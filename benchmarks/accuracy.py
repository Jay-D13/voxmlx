"""Word and character error rates of voxmlx on a manifest of recordings.

    accuracy.py synth --out DIR
    accuracy.py run --manifest M.jsonl --model DIR [--config mode=stream,delay_ms=240 ...]

A manifest has one JSON object per line: {"id", "audio", "text"}. Relative audio
paths are relative to the manifest's folder.

A config is comma-separated key=value pairs: mode (file or stream, default stream)
and any of delay_ms, audio_batch_ms, context_size. Only the settings given are
passed to voxmlx, so its own defaults and checks apply.

mode=file calls voxmlx.transcribe. mode=stream runs the live loop
voxmlx.stream.stream_transcribe on the recording (see replay_stream).

Output is JSON lines on stdout: one per recording, then one summary per config.
"""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unicodedata
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Measure this checkout.

import mlx.core as mx
import numpy as np
import soundfile as sf
import voxmlx
import voxmlx.stream as stream
from voxmlx.audio import SAMPLES_PER_TOKEN, load_audio

SENTENCES = [
    "Bonjour, je m'appelle Claire et j'habite à Lyon depuis 2015.",
    'Il y a 80 personnes dans la salle ce soir.',
    'Le rendez-vous est fixé au 21 mars à la mairie de Saint-Étienne.',
    "Aujourd'hui, l'économie française a progressé de 3,5 % en un an.",
    'Où est-ce que tu as mis les clés de la voiture ?',
    'Les élèves ont lu « Les Misérables » de Victor Hugo cet été.',
    'Ma chambre porte le numéro 1 204, au douzième étage.',
    'Il fait très froid à Strasbourg ce matin.',
    'Nous avons commandé 97 croissants pour le petit-déjeuner.',
    "Le cœur de la ville bat au rythme des festivals d'été.",
    'Jean-Pierre et Marie-Hélène se sont mariés en 1999.',
    "Pouvez-vous m'indiquer le chemin de la gare, s'il vous plaît ?",
    'La tour Eiffel mesure environ 330 mètres de haut.',
    "C'est la 2e fois cette semaine que le train a du retard.",
    'Le 1er mai, les magasins sont fermés dans toute la France.',
    "Après avoir traversé la Bretagne à vélo pendant trois semaines, en dormant chez "
    "l'habitant et en goûtant les crêpes de chaque village, Thomas a décidé qu'il "
    "reviendrait l'année prochaine avec ses deux sœurs et leurs enfants.",
    "Qu'est-ce qu'on mange ce soir ? Des pâtes ou du poisson ?",
    'Le musée du Louvre accueille près de 9 000 visiteurs par jour.',
    'Elle a payé 45 euros pour deux billets de théâtre.',
    'Il était une fois, dans un petit château près de Bordeaux, une princesse très curieuse.',
    'Les 71 participants venaient de Genève, de Bruxelles et de Montréal.',
]


def normalize(text):
    """Unicode NFKC, lowercase, punctuation and symbols (apostrophes, hyphens
    included) to spaces, whitespace collapsed. Accents are kept.

    Numbers or accents written differently ("80" and "quatre-vingts", "Etienne"
    and "Étienne") count as errors.
    """
    text = unicodedata.normalize('NFKC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in 'PS' else c for c in text).split())


def edit_distance(ref, hyp):
    """Levenshtein distance between two sequences (word lists or strings)."""
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        diagonal, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (r != h))
    return row[-1]


def read_manifest(path):
    """Manifest rows with audio paths made absolute; fails if a file is missing."""
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        row['audio'] = str((path.parent / row['audio']).resolve())
        if not Path(row['audio']).is_file():
            raise FileNotFoundError(f"Audio for {row['id']} not found: {row['audio']}")
    return rows


def replay_stream(loaded, audio, **settings):
    """Run voxmlx.stream.stream_transcribe on a recording and return its text.

    Only the microphone, the condition's wait, and load_model are replaced: each
    wait delivers the next 80 ms block at once (no real-time pacing), and the end
    of the recording raises KeyboardInterrupt, the live Ctrl+C path with its flush.
    """
    blocks = [audio[i:i + SAMPLES_PER_TOKEN] for i in range(0, len(audio), SAMPLES_PER_TOKEN)]
    mic = {}

    class Microphone:
        def __init__(self, callback, **kwargs):
            mic['callback'] = callback
        start = stop = close = lambda self: None

    class Condition(threading.Condition):
        def wait(self, timeout=None):
            if not blocks:
                raise KeyboardInterrupt
            block = blocks.pop(0)
            mic['callback'](block[:, None], len(block), None, None)
            return True

    pieces = []
    with patch.object(stream, 'load_model', return_value=loaded), \
            patch.object(stream.sd, 'InputStream', Microphone), \
            patch.object(stream, 'threading', SimpleNamespace(Condition=Condition)), \
            redirect_stdout(io.StringIO()):  # Hide "Listening...".
        stream.stream_transcribe(on_text=pieces.append, **settings)
    return ''.join(pieces)


def transcribe(loaded, model_path, mode, settings, audio_path):
    if mode == 'file':
        with patch.object(voxmlx, 'load_model', return_value=loaded):
            return voxmlx.transcribe(audio_path, model_path=model_path, **settings)
    return replay_stream(loaded, load_audio(audio_path), model_path=model_path, **settings)


def parse_config(spec):
    """Parse "mode=stream,delay_ms=240" into ('stream', {'delay_ms': 240})."""
    settings = dict(item.split('=', 1) for item in spec.split(','))
    mode = settings.pop('mode', 'stream')
    if mode not in ('file', 'stream'):
        raise ValueError(f'mode must be file or stream, not {mode!r}')
    return mode, {key: int(value) for key, value in settings.items()}


def run(manifest, model_path, specs):
    rows = read_manifest(manifest)
    loaded = voxmlx.load_model(model_path)
    configs = [(spec, *parse_config(spec)) for spec in specs]
    # Untimed first pass in each mode, so compilation is not counted in wall time.
    for mode, settings in {mode: settings for _, mode, settings in configs}.items():
        transcribe(loaded, model_path, mode, settings, rows[0]['audio'])
    for spec, mode, settings in configs:
        totals = dict(word_errors=0, reference_words=0, char_errors=0, reference_chars=0)
        wall = 0.0
        for row in rows:
            start = time.perf_counter()
            hypothesis = transcribe(loaded, model_path, mode, settings, row['audio']).strip()
            wall += time.perf_counter() - start
            ref, hyp = normalize(row['text']), normalize(hypothesis)
            counts = dict(word_errors=edit_distance(ref.split(), hyp.split()), reference_words=len(ref.split()),
                          char_errors=edit_distance(ref, hyp), reference_chars=len(ref))
            totals = {key: totals[key] + counts[key] for key in totals}
            print(json.dumps({'config': spec, 'id': row['id'], 'reference': row['text'], 'hypothesis': hypothesis,
                              'word_errors': counts['word_errors'], 'reference_words': counts['reference_words']},
                             ensure_ascii=False), flush=True)
        print(json.dumps({
            'config': spec, 'recordings': len(rows),
            'wer': totals['word_errors'] / totals['reference_words'],
            'cer': totals['char_errors'] / totals['reference_chars'], **totals,
            'wall_seconds': round(wall, 2),
            'audio_seconds': round(sum(sf.info(row['audio']).duration for row in rows), 2),
            'date': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'mlx': mx.__version__, 'model': model_path}), flush=True)


def synth(out):
    """Synthetic French test set from macOS `say -v Thomas`, 16 kHz WAVs.

    Writes manifest.jsonl (one clip per sentence) and manifest-long.jsonl (one
    recording of all clips joined with 0.8 s of silence).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, pieces = [], []
    for i, text in enumerate(SENTENCES, 1):
        name = f'synthetic-thomas-{i:02d}.wav'
        subprocess.run(['say', '-v', 'Thomas', '-o', str(out / name), '--file-format=WAVE',
                        '--data-format=LEI16@16000'], input=text, text=True, check=True)
        rows.append({'id': name[:-4], 'audio': name, 'text': text, 'synthetic': True})
        audio, rate = sf.read(out / name, dtype='float32')
        pieces += [audio, np.zeros(int(0.8 * rate), np.float32)]
    sf.write(out / 'synthetic-thomas-long.wav', np.concatenate(pieces[:-1]), rate, subtype='PCM_16')
    long_row = {'id': 'synthetic-thomas-long', 'audio': 'synthetic-thomas-long.wav',
                'text': ' '.join(SENTENCES), 'synthetic': True}
    for name, manifest_rows in (('manifest.jsonl', rows), ('manifest-long.jsonl', [long_row])):
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifest_rows),
                                encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('synth').add_argument('--out', required=True)
    bench = commands.add_parser('run')
    bench.add_argument('--manifest', required=True)
    bench.add_argument('--model', required=True, help='Local model directory')
    bench.add_argument('--config', action='append', help='Repeatable; default: mode=file and mode=stream')
    args = parser.parse_args(argv)
    if args.command == 'synth':
        synth(args.out)
    elif not Path(args.model, 'config.json').is_file():
        parser.error(f'Not a local model directory: {args.model}')  # load_model would download it.
    else:
        run(args.manifest, args.model, args.config or ['mode=file', 'mode=stream'])


if __name__ == '__main__':
    main()
