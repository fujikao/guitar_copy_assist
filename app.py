import os
import gc
import json
import uuid
import base64
from io import BytesIO
from datetime import datetime

os.environ['MPLCONFIGDIR'] = '/tmp'

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'guitar_copy_assist_secret_key'

UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_DURATION = 360.0
MAX_SAVED_SONGS = 100
CHORDS_DICT_FILE = 'chords_dict.json'
SAVED_SONGS_FILE = 'saved_songs.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
EASY_OPEN_CHORDS = {'C', 'G', 'D', 'A', 'E', 'Am', 'Em', 'Dm'}

def load_chords_dict():
    if os.path.exists(CHORDS_DICT_FILE):
        with open(CHORDS_DICT_FILE, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    return {}

CHORDS_DICT = load_chords_dict()

def load_saved_songs():
    if not os.path.exists(SAVED_SONGS_FILE):
        with open(SAVED_SONGS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"songs": []}, f, ensure_ascii=False, indent=2)
    try:
        with open(SAVED_SONGS_FILE, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return None

def save_songs_data(data):
    with open(SAVED_SONGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1).lower() in ALLOWED_EXTENSIONS
def analyze_audio(file_path):
    y, sr = librosa.load(file_path, sr=11025, mono=True, dtype=np.float32)
    duration = float(len(y)) / sr
    if duration > MAX_DURATION or len(y) == 0 or np.max(np.abs(y)) < 1e-4:
        del y
        gc.collect()
        raise ValueError("ERR-004")

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(np.atleast_1d(tempo))))
    if bpm <= 0:
        bpm = 120

    hop_length = 1024
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    del y
    gc.collect()

    chroma_mean = np.mean(chroma, axis=1)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_score = -1
    estimated_key = "C Major"
    for i in range(12):
        maj_corr = np.corrcoef(chroma_mean, np.roll(major_profile, i))
        min_corr = np.corrcoef(chroma_mean, np.roll(minor_profile, i))
        if maj_corr > best_score:
            best_score = maj_corr
            estimated_key = f"{NOTES[i]} Major"
        if min_corr > best_score:
            best_score = min_corr
            estimated_key = f"{NOTES[i]} Minor"

    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4
    total_bars = max(1, int(duration // bar_duration))
    analyze_bars = min(total_bars, 64)

    templates = {}
    for i, note in enumerate(NOTES):
        vec_maj = np.zeros(12)
        vec_maj[[i, (i + 4) % 12, (i + 7) % 12]] = 1.0
        templates[note] = vec_maj
        vec_min = np.zeros(12)
        vec_min[[i, (i + 3) % 12, (i + 7) % 12]] = 1.0
        templates[f"{note}m"] = vec_min

    original_chords = []
    for b in range(analyze_bars):
        start_sample = int(b * bar_duration * sr / hop_length)
        end_sample = int((b + 1) * bar_duration * sr / hop_length)
        if start_sample >= chroma.shape:
            break
        bar_chroma = np.mean(chroma[:, start_sample:max(start_sample + 1, end_sample)], axis=1)
        if np.linalg.norm(bar_chroma) < 1e-5:
            original_chords.append("不明")
            continue
        best_chord = "C"
        max_sim = -1
        for chord_name, tmpl in templates.items():
            sim = np.dot(bar_chroma, tmpl) / (np.linalg.norm(bar_chroma) * np.linalg.norm(tmpl))
            if sim > max_sim:
                max_sim = sim
                best_chord = chord_name
        original_chords.append(best_chord)

    del chroma
    gc.collect()
    return estimated_key, bpm, original_chords, len([c for c in original_chords if c != "不明"]) == 0
def transpose_chord(chord, capo):
    if chord == "不明" or capo == 0:
        return chord
    is_minor = chord.endswith('m')
    root = chord[:-1] if is_minor else chord
    if root not in NOTES:
        return chord
    return f"{NOTES[(NOTES.index(root) - capo) % 12]}{'m' if is_minor else ''}"

def convert_chords_with_capo(original_chords, capo):
    converted = [transpose_chord(c, capo) for c in original_chords]
    return converted, any(c != "不明" and c not in CHORDS_DICT for c in converted)

def find_best_capo(original_chords):
    best_capo, best_score = 0, -9999
    for capo in range(8):
        score = sum(10 if c in EASY_OPEN_CHORDS else (2 if c in CHORDS_DICT else -5) for c in [transpose_chord(x, capo) for x in original_chords]) - capo * 0.2
        if score > best_score:
            best_score, best_capo = score, capo
    return best_capo

def generate_chord_diagram_base64(chord_name):
    if chord_name not in CHORDS_DICT:
        return None
    try:
        info = CHORDS_DICT[chord_name]
        frets, base_fret = info["frets"], info.get("base_fret", 1)
        fig, ax = plt.subplots(figsize=(2.0, 2.4), dpi=90)
        fig.patch.set_facecolor('#f9f1e4')
        ax.set_facecolor('#f9f1e4')
        for s in range(6):
            ax.plot([s, s],, color='#3e2723', linewidth=1.3)
        for f in range(5):
            ax.plot(, [f, f], color='#3e2723', linewidth=(3.5 if (f == 0 and base_fret == 1) else 1.2))
        for s_idx, fret_val in enumerate(frets):
            if fret_val == 'x':
                ax.text(s_idx, -0.35, '×', ha='center', va='center', fontsize=11, color='#b71c1c', fontweight='bold')
            elif fret_val == 0:
                ax.text(s_idx, -0.35, '○', ha='center', va='center', fontsize=10, color='#2e7d32', fontweight='bold')
            elif 1 <= (fret_val - base_fret + 1) <= 4:
                ax.add_patch(plt.Circle((s_idx, fret_val - base_fret + 0.5), 0.24, color='#5d4037', zorder=5))
        if base_fret > 1:
            ax.text(-0.55, 0.5, f"{base_fret}fr", ha='right', va='center', fontsize=9, color='#3e2723', fontweight='bold')
        ax.set_title(chord_name, fontsize=13, fontweight='bold', color='#3e2723', pad=14)
        ax.set_xlim(-0.8, 5.5)
        ax.set_ylim(4.3, -0.8)
        ax.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return "ERROR"

def build_diagrams(chords):
    diagrams, err = {}, False
    for c in dict.fromkeys(c for c in chords if c != "不明"):
        img = generate_chord_diagram_base64(c)
        if img == "ERROR":
            err = True
        elif img:
            diagrams[c] = img
    gc.collect()
    return diagrams, err
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio_file' not in request.files or request.files['audio_file'].filename == '':
        flash("解析する音声ファイルを選択してください。", "error")
        return redirect(url_for('index'))
    file = request.files['audio_file']
    if not allowed_file(file.filename):
        flash("MP3またはWAV形式の音声ファイルを選択してください。", "error")
        return redirect(url_for('index'))
    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_FILE_SIZE:
        flash("ファイルサイズが大きすぎます。20MB以下のファイルを使用してください。", "error")
        return redirect(url_for('index'))
    file.seek(0)
    temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}_{file.filename}")
    try:
        file.save(temp_path)
        estimated_key, bpm, original_chords, warn_err006 = analyze_audio(temp_path)
        if warn_err006:
            flash("コード進行を正しく解析できませんでした。音量や音質の異なる音源で再度お試しください。", "warning")
        best_capo = find_best_capo(original_chords)
        converted_chords, warn_err007 = convert_chords_with_capo(original_chords, best_capo)
        if warn_err007:
            flash("一部のコードを簡単な押さえ方に変換できませんでした。原曲のコードを表示しています。", "warning")
        diagrams, warn_err008 = build_diagrams(converted_chords)
        if warn_err008:
            flash("コードの図を表示できませんでした。コード名を確認して演奏してください。", "warning")
        return render_template('result.html', song_id=None, title=os.path.splitext(file.filename), key=estimated_key, bpm=bpm, best_capo=best_capo, selected_capo=best_capo, original_chords=original_chords, converted_chords=converted_chords, diagrams=diagrams, mode='beginner')
    except ValueError as ve:
        flash("6分以内の音声が含まれる楽曲ファイルを選択してください。" if str(ve) == "ERR-004" else "音声ファイルの読み込みに失敗しました。", "error")
        return redirect(url_for('index'))
    except Exception:
        flash("音声ファイルの読み込みに失敗しました。別のファイルでお試しください。", "error")
        return redirect(url_for('index'))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()

@app.route('/analyze/capo', methods=['POST'])
def analyze_capo():
    title, key, bpm = request.form.get('title', ''), request.form.get('key', 'C Major'), int(request.form.get('bpm', 120))
    best_capo, selected_capo, mode = int(request.form.get('best_capo', 0)), int(request.form.get('selected_capo', 0)), request.form.get('mode', 'beginner')
    original_chords = json.loads(request.form.get('original_chords', '[]'))
    converted_chords, _ = convert_chords_with_capo(original_chords, 0 if mode == 'original' else selected_capo)
    diagrams, _ = build_diagrams(converted_chords)
    return render_template('result.html', song_id=None, title=title, key=key, bpm=bpm, best_capo=best_capo, selected_capo=selected_capo, original_chords=original_chords, converted_chords=converted_chords, diagrams=diagrams, mode=mode)

@app.route('/songs/save', methods=['POST'])
def save_song():
    title = request.form.get('title', '').strip()
    key, bpm, best_capo, selected_capo = request.form.get('key', 'C Major'), int(request.form.get('bpm', 120)), int(request.form.get('best_capo', 0)), int(request.form.get('selected_capo', 0))
    original_chords, converted_chords = json.loads(request.form.get('original_chords', '[]')), json.loads(request.form.get('converted_chords', '[]'))
    if not title:
        flash("保存する楽曲タイトルを入力してください。", "error")
        diagrams, _ = build_diagrams(converted_chords)
        return render_template('result.html', song_id=None, title="", key=key, bpm=bpm, best_capo=best_capo, selected_capo=selected_capo, original_chords=original_chords, converted_chords=converted_chords, diagrams=diagrams, mode='beginner')
    data = load_saved_songs()
    if len(data["songs"]) >= MAX_SAVED_SONGS:
        flash("保存できる楽曲数の上限に達しています。", "error")
        return redirect(url_for('codebook'))
    data["songs"].insert(0, {"song_id": f"song_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}", "title": title, "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "key": key, "bpm": bpm, "best_capo": best_capo, "selected_capo": selected_capo, "original_chords": original_chords, "converted_chords": converted_chords})
    save_songs_data(data)
    flash(f"「{title}」をマイコードブックに保存しました。", "success")
    return redirect(url_for('codebook'))

@app.route('/codebook', methods=['GET'])
def codebook():
    data = load_saved_songs()
    return render_template('codebook.html', songs=data.get("songs", []) if data else [])

@app.route('/songs/<song_id>', methods=['GET'])
def open_song(song_id):
    data = load_saved_songs()
    song = next((s for s in data["songs"] if s["song_id"] == song_id), None) if data else None
    if not song:
        flash("指定された楽曲データが見つかりませんでした。", "error")
        return redirect(url_for('codebook'))
    diagrams, _ = build_diagrams(song["converted_chords"])
    return render_template('result.html', song_id=song["song_id"], title=song["title"], key=song["key"], bpm=song["bpm"], best_capo=song.get("best_capo", song["selected_capo"]), selected_capo=song["selected_capo"], original_chords=song["original_chords"], converted_chords=song["converted_chords"], diagrams=diagrams, mode='beginner')

@app.route('/songs/<song_id>/capo', methods=['POST'])
def update_saved_song_capo(song_id):
    data = load_saved_songs()
    song = next((s for s in data["songs"] if s["song_id"] == song_id), None) if data else None
    if not song:
        flash("指定された楽曲データが見つかりませんでした。", "error")
        return redirect(url_for('codebook'))
    selected_capo, mode, title = int(request.form.get('selected_capo', 0)), request.form.get('mode', 'beginner'), request.form.get('title', song["title"])
    converted_chords, _ = convert_chords_with_capo(song["original_chords"], 0 if mode == 'original' else selected_capo)
    diagrams, _ = build_diagrams(converted_chords)
    return render_template('result.html', song_id=song_id, title=title, key=song["key"], bpm=song["bpm"], best_capo=song.get("best_capo", selected_capo), selected_capo=selected_capo, original_chords=song["original_chords"], converted_chords=converted_chords, diagrams=diagrams, mode=mode)

@app.route('/songs/<song_id>/update', methods=['POST'])
def overwrite_song(song_id):
    title = request.form.get('title', '').strip()
    if not title:
        flash("保存する楽曲タイトルを入力してください。", "error")
        return redirect(url_for('open_song', song_id=song_id))
    data = load_saved_songs()
    song = next((s for s in data["songs"] if s["song_id"] == song_id), None) if data else None
    if song:
        selected_capo = int(request.form.get('selected_capo', 0))
        converted_chords, _ = convert_chords_with_capo(song["original_chords"], selected_capo)
        song.update({"title": title, "selected_capo": selected_capo, "converted_chords": converted_chords, "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')})
        save_songs_data(data)
        flash(f"「{title}」の設定を上書き保存しました。", "success")
    return redirect(url_for('open_song', song_id=song_id))

@app.route('/songs/<song_id>/delete', methods=['POST'])
def delete_song(song_id):
    data = load_saved_songs()
    if data:
        data["songs"] = [s for s in data["songs"] if s["song_id"] != song_id]
        save_songs_data(data)
        flash("楽曲を削除しました。", "success")
    return redirect(url_for('codebook'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
