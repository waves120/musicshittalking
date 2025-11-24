import sys
import os
import zipfile
from datetime import datetime
import uuid

import torch
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

from transformers import AutoProcessor, MusicgenForConditionalGeneration
from scipy.io import wavfile


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Загружаем твой .ui
        uic.loadUi("music_generator_ui.ui", self)

        # Плеер
        self.player = QMediaPlayer(self)
        self.player.setVolume(100)

        # Модель MusicGen (ленивая инициализация)
        self.processor = None
        self.model = None
        self.device = "cpu"

        # Последние данные
        self.last_prompt = None
        self.last_sr = None
        self.last_loop_audio = None     # int16 [samples, channels]
        self.last_full_audio = None
        self.last_enh_audio = None

        # Текущие имена файлов (определяются при генерации)
        self.loop_path = None
        self.full_path = None
        self.enh_path = None

        # Стартовое состояние
        self.btnStop.setEnabled(False)
        self.btnDownloadZip.setEnabled(False)
        self.btnEnhanceCloud.setEnabled(False)
        self.progressBar.setValue(0)
        self.labelStatus.setText("Готов к генерации")

        # Сигналы
        self.btnGenerate.clicked.connect(self.on_generate_clicked)
        self.btnStop.clicked.connect(self.on_stop_clicked)
        self.btnDownloadZip.clicked.connect(self.on_download_zip_clicked)
        self.btnEnhanceCloud.clicked.connect(self.on_enhance_clicked)

    # ---------- ВСПОМОГАТЕЛЬНОЕ ----------

    def log(self, text: str):
        """Лог в txtLog + статусбар."""
        try:
            self.txtLog.appendPlainText(text)
        except Exception:
            pass
        self.statusBar().showMessage(text)
        self.labelStatus.setText(text)
        QtWidgets.QApplication.processEvents()

    def _init_music_model(self):
        """Ленивая загрузка процессора и модели MusicGen."""
        if self.model is None or self.processor is None:
            self.log("Загрузка модели MusicGen (facebook/musicgen-small)...")
            QtWidgets.QApplication.processEvents()

            self.processor = AutoProcessor.from_pretrained(
                "facebook/musicgen-small"
            )
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                "facebook/musicgen-small"
            )
            # На CPU
            self.model.to(self.device)

            self.log("Модель MusicGen загружена")

    def _build_prompt(self) -> str:
        """Собрать текстовый промпт из полей UI."""
        genre = self.comboGenre.currentText().strip()
        key = self.comboKey.currentText().strip()
        bpm = int(self.spinBPM.value())

        instruments = []
        if self.chkDrums.isChecked():
            instruments.append("drums")
        if self.chkBass.isChecked():
            instruments.append("bass")
        if self.chkPiano.isChecked():
            instruments.append("piano")
        if self.chkGuitar.isChecked():
            instruments.append("guitar")
        if self.chkStrings.isChecked():
            instruments.append("strings")
        if self.chkPads.isChecked():
            instruments.append("synth pads")

        parts = []
        if genre:
            parts.append(f"{genre} style")
        if key:
            parts.append(f"in the key of {key}")
        if bpm > 0:
            parts.append(f"at {bpm} BPM")

        base_desc = ", ".join(parts) if parts else "instrumental music"
        instr_desc = " with " + " and ".join(instruments) if instruments else ""

        full_prompt = f"{base_desc}{instr_desc}, high quality, 4 bar loop, instrumental"
        return full_prompt

    def _prepare_file_names(self):
        """
        Генерируем уникальные имена файлов для текущей генерации.
        Пример: music_20251116_211530_123456_abcd1234_loop.wav
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        uid = uuid.uuid4().hex[:8]
        base = os.path.abspath(f"music_{stamp}_{uid}")

        self.loop_path = base + "_loop.wav"
        self.full_path = base + "_full.wav"
        self.enh_path = base + "_full_enh.wav"

    # ---------- ГЕНЕРАЦИЯ ----------

    def on_generate_clicked(self):
        """Кнопка 'Сгенерировать'."""
        bpm = int(self.spinBPM.value())
        if bpm <= 0:
            bpm = 120
            self.spinBPM.setValue(bpm)

        total_duration_sec = int(self.spinDuration.value())
        if total_duration_sec <= 0:
            total_duration_sec = 8
            self.spinDuration.setValue(total_duration_sec)

        full_prompt = self._build_prompt()
        self.last_prompt = full_prompt
        self.log(f"Промпт: {full_prompt}")

        # 4 такта 4/4 = 16 ударов
        beats_per_second = bpm / 60.0
        loop_duration_sec = 16.0 / beats_per_second

        # оценка кол-ва токенов: MusicGen ~50 токенов/сек
        approx_tokens = int(loop_duration_sec * 50)
        max_new_tokens = int(np.clip(approx_tokens, 64, 1024))

        # Блокируем кнопки
        self.btnGenerate.setEnabled(False)
        self.btnStop.setEnabled(False)
        self.btnDownloadZip.setEnabled(False)
        self.btnEnhanceCloud.setEnabled(False)
        self.progressBar.setValue(10)

        # Инициализируем модель
        try:
            self._init_music_model()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка модели", f"Не удалось загрузить MusicGen:\n{e}"
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            return

        # Подготовка входа
        self.log("Подготовка входных данных...")
        try:
            inputs = self.processor(
                text=[full_prompt],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка препроцессора", f"Не удалось подготовить вход:\n{e}"
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            return

        # Генерация
        self.log(
            f"Генерация 4-тактового фрагмента "
            f"(max_new_tokens={max_new_tokens})..."
        )
        try:
            with torch.no_grad():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка генерации", f"Не удалось сгенерировать музыку:\n{e}"
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            self.log("Ошибка генерации")
            return

        self.progressBar.setValue(60)
        self.log("Постобработка лупа...")

        # audio_values: [batch, channels, samples]
        try:
            audio_tensor = audio_values[0, 0]  # mono
            audio = audio_tensor.detach().cpu().numpy()
            sr = self.model.config.audio_encoder.sampling_rate
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка аудиоданных", f"Не удалось извлечь аудио:\n{e}"
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            self.log("Ошибка аудиоданных")
            return

        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]  # [samples] -> [samples, 1]

        # ---- делаем луп ≈ 4 такта (без безумного tile) ----
        target_loop_samples = int(loop_duration_sec * sr)
        current_samples = audio.shape[0]

        if current_samples >= target_loop_samples:
            audio_loop = audio[:target_loop_samples, :]
        else:
            # если немного короче – просто используем как есть
            audio_loop = audio

        loop_len = audio_loop.shape[0]
        loop_seconds = loop_len / sr

        # защита от совсем короткого мусора
        if loop_len <= 0 or loop_seconds < 1.0:
            self.log(
                f"Слишком короткий луп ({loop_seconds:.3f} с) – генерация неудачна."
            )
            QtWidgets.QMessageBox.warning(
                self,
                "Неудачная генерация",
                "Модель вернула слишком короткий фрагмент (<1 секунды).\n"
                "Попробуйте ещё раз или измените параметры (жанр/BPM).",
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            return

        # нормализация лупа
        max_val = float(np.max(np.abs(audio_loop)))
        if max_val < 1e-6:
            max_val = 1.0
        audio_loop_norm = (audio_loop / max_val) * 0.99
        loop_int16 = (audio_loop_norm * 32767).astype(np.int16)

        # ---- строим полный трек нужной длительности ----
        self.log("Сборка полного трека из лупа...")
        target_full_samples = int(total_duration_sec * sr)
        loop_len = loop_int16.shape[0]

        if loop_len <= 0:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка генерации", "Пустой аудиолуп, генерация не удалась."
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            self.log("Пустой луп")
            return

        # разумный лимит повторов
        max_reps = 10000
        reps_full = int(np.ceil(target_full_samples / loop_len))
        if reps_full > max_reps:
            reps_full = max_reps
            target_full_samples = loop_len * max_reps
            self.log(
                "Длительность ограничена из-за очень большого числа повторов "
                f"(reps={reps_full})."
            )

        full_audio = np.tile(loop_int16, (reps_full, 1))[:target_full_samples, :]

        # генерируем уникальные пути для файлов
        self._prepare_file_names()

        # сохраняем WAV
        try:
            wavfile.write(self.loop_path, sr, loop_int16)
            wavfile.write(self.full_path, sr, full_audio)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось сохранить WAV:\n{e}"
            )
            self.btnGenerate.setEnabled(True)
            self.progressBar.setValue(0)
            self.log("Ошибка сохранения WAV")
            return

        # сохраняем в атрибутах
        self.last_sr = sr
        self.last_loop_audio = loop_int16
        self.last_full_audio = full_audio
        self.last_enh_audio = None  # сбрасываем, так как новая генерация

        # готовим плеер
        self.player.stop()
        media = QMediaContent(QUrl.fromLocalFile(self.full_path))
        self.player.setMedia(media)

        self.progressBar.setValue(100)
        self.btnGenerate.setEnabled(True)
        self.btnStop.setEnabled(True)
        self.btnDownloadZip.setEnabled(True)
        self.btnEnhanceCloud.setEnabled(True)

        self.log(
            f"Готово. Луп: {os.path.basename(self.loop_path)} (~{loop_seconds:.2f} с), "
            f"Трек: {os.path.basename(self.full_path)} (≈{target_full_samples / sr:.1f} с)"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Готово",
            f"Сгенерирован 4-тактовый луп (~{loop_seconds:.2f} с) и трек.",
        )

        # авто-плей полного трека
        self.player.setPosition(0)
        self.player.play()

    # ---------- ПРОИГРЫВАНИЕ / СТОП ----------

    def on_stop_clicked(self):
        self.player.stop()
        self.log("Воспроизведение остановлено")

    # ---------- ZIP ----------

    def on_download_zip_clicked(self):
        if self.last_full_audio is None or self.last_sr is None:
            QtWidgets.QMessageBox.warning(
                self, "Нет данных", "Сначала сгенерируйте трек."
            )
            return

        default_name = f"music_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Сохранить ZIP архив",
            default_name,
            "ZIP архивы (*.zip)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".zip"):
            filename += ".zip"

        try:
            with zipfile.ZipFile(filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                if self.loop_path and os.path.exists(self.loop_path):
                    zf.write(self.loop_path, arcname=os.path.basename(self.loop_path))
                if self.full_path and os.path.exists(self.full_path):
                    zf.write(self.full_path, arcname=os.path.basename(self.full_path))
                if self.enh_path and os.path.exists(self.enh_path):
                    zf.write(self.enh_path, arcname=os.path.basename(self.enh_path))

                meta_lines = []
                if self.last_prompt:
                    meta_lines.append(f"Prompt: {self.last_prompt}")
                if self.last_sr:
                    meta_lines.append(f"Sample rate: {self.last_sr} Hz")
                meta = "\n".join(meta_lines) or "No metadata"
                zf.writestr("metadata.txt", meta)

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка ZIP", f"Не удалось сохранить ZIP архив:\n{e}"
            )
            return

        self.log(f"ZIP сохранён: {filename}")
        QtWidgets.QMessageBox.information(
            self, "ZIP сохранён", f"Архив успешно сохранён:\n{filename}"
        )

    # ---------- "УЛУЧШИТЬ В ОБЛАКЕ" ----------

    def on_enhance_clicked(self):
        if self.last_full_audio is None or self.last_sr is None:
            QtWidgets.QMessageBox.warning(
                self, "Нет данных", "Сначала сгенерируйте трек."
            )
            return

        self.log("Имитация улучшения в облаке (мастеринг)...")
        self.progressBar.setValue(30)

        audio = self.last_full_audio.astype(np.float32) / 32767.0

        # простая "облачная магия": мягкий сатурационный компрессор + нормализация
        enhanced = np.tanh(audio * 1.5)
        max_val = float(np.max(np.abs(enhanced)))
        if max_val < 1e-6:
            max_val = 1.0
        enhanced = (enhanced / max_val) * 0.99

        enh_int16 = (enhanced * 32767).astype(np.int16)

        # Если по какой-то причине имя ещё не задано — сделаем его от full_path
        if self.enh_path is None:
            if self.full_path:
                root, ext = os.path.splitext(self.full_path)
                self.enh_path = root + "_enh.wav"
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                uid = uuid.uuid4().hex[:8]
                base = os.path.abspath(f"music_{stamp}_{uid}")
                self.enh_path = base + "_full_enh.wav"

        try:
            wavfile.write(self.enh_path, self.last_sr, enh_int16)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось сохранить улучшенный WAV:\n{e}"
            )
            self.progressBar.setValue(0)
            self.log("Ошибка улучшения")
            return

        self.last_enh_audio = enh_int16

        # Переключаем плеер на улучшенный трек
        self.player.stop()
        media = QMediaContent(QUrl.fromLocalFile(self.enh_path))
        self.player.setMedia(media)

        self.progressBar.setValue(100)
        self.log(f"Улучшенная версия сохранена: {os.path.basename(self.enh_path)}")
        QtWidgets.QMessageBox.information(
            self,
            "Готово",
            "Трек улучшен (имитация облака) и сохранён как отдельный файл.",
        )

        self.player.setPosition(0)
        self.player.play()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
