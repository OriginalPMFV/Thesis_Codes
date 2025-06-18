from __future__ import annotations

import sys
import subprocess
import threading
import time
import signal
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

# ─────────────────── Configuração ────────────────────
# Caminhos dos scripts que podem ser iniciados pela GUI
SCRIPT_CAMERA = Path("camera_test.py")            # teste de câmara
SCRIPT_MAIN   = Path("vision_script.py")          # detecção + controlo UR3e

# Argumentos opcionais a passar aos scripts
SCRIPT_CAMERA_ARGS: list[str] = []
SCRIPT_MAIN_ARGS:   list[str] = []

# ─────────────────── Classe GUI ──────────────────────
class UR3eGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Colheita UR3e – Controle")
        self.geometry("700x460")
        self.resizable(False, False)

        # Variáveis de estado de execução
        self.process: subprocess.Popen[str] | None = None
        self.start_time: float | None = None
        self.running_script: str | None = None

        self._build_widgets()     # constrói os componentes da interface
        self._update_timer()      # inicia contador de tempo

    # ─────────────── criação de widgets ───────────────
    def _build_widgets(self):
        style = ttk.Style(self)
        style.configure("TButton", font=("Segoe UI", 11))
        style.configure("Status.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Timer.TLabel", font=("Consolas", 11))

        frame_btns = ttk.Frame(self, padding=20)
        frame_btns.pack(fill="x")

        # Botão para testar a câmara
        self.btn_camera = ttk.Button(
            frame_btns,
            text="Testar Câmara",
            command=self._start_camera,
            width=15,
        )
        self.btn_camera.pack(side="left", expand=True, padx=10)

        # Botão para iniciar a deteção e controlo
        self.btn_start = ttk.Button(
            frame_btns,
            text="START",
            command=self._start_main,
            width=15,
        )
        self.btn_start.pack(side="left", expand=True, padx=10)

        # Botão para parar o processo em execução
        self.btn_stop = ttk.Button(
            frame_btns,
            text="STOP",
            command=self._stop_process,
            width=15,
            state="disabled",  # só habilitado quando um script estiver a correr
        )
        self.btn_stop.pack(side="left", expand=True, padx=10)

        # ─── Estado + temporizador ───
        self.status_var = tk.StringVar(value="Pronto")     # estado atual
        self.timer_var = tk.StringVar(value="00:00:00")    # tempo decorrido

        ttk.Label(self, textvariable=self.status_var, style="Status.TLabel").pack(pady=(5, 0))
        ttk.Label(self, textvariable=self.timer_var, style="Timer.TLabel").pack()

        # ─── Caixa de log ───
        self.txt_log = scrolledtext.ScrolledText(self, height=17, state="disabled", font=("Consolas", 9))
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=10)

    # ─────────────── logging helper ───────────────
    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", f"{timestamp} | {msg}\n")
        self.txt_log.configure(state="disabled")
        self.txt_log.yview_moveto(1.0)  # scroll automático

    # ─────────────── Botões ───────────────
    def _start_camera(self):
        # Inicia script de teste da câmara
        self._launch_process(SCRIPT_CAMERA, "Câmara em execução…", SCRIPT_CAMERA_ARGS, show_timer=False)

    def _start_main(self):
        # Inicia script principal (visão + UR3e)
        self._launch_process(SCRIPT_MAIN, "Colheita em execução…", SCRIPT_MAIN_ARGS, show_timer=True)

    # ─────────────── Execução de scripts ───────────────
    def _launch_process(self, script_path: Path, status_text: str, args: list[str], *, show_timer: bool):
        if self.process:
            messagebox.showwarning("Processo já em execução", "Termina ou espera antes de iniciar outro script.")
            return
        if not script_path.exists():
            messagebox.showerror("Script não encontrado", f"{script_path} não existe!")
            return

        # Atualiza estado da interface
        self.running_script = script_path.name
        self.status_var.set(status_text)
        self.start_time = time.time() if show_timer else None
        self.btn_stop.config(state="normal")
        self._log(f"Iniciando {script_path} {args if args else ''}")

        def target():
            try:
                # Cria subprocesso com flags específicas para Windows (permite terminar com sinal)
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0
                self.process = subprocess.Popen(
                    [sys.executable, str(script_path), *args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                )
                # Lê e mostra stdout em tempo real
                assert self.process.stdout is not None
                for line in self.process.stdout:
                    self._log(line.rstrip())
                self.process.wait()
            except Exception as exc:
                self._log(f"❌ Erro: {exc}")
            finally:
                self.after(0, self._on_process_end)

        threading.Thread(target=target, daemon=True).start()  # corre subprocesso em paralelo

    def _on_process_end(self):
        # Quando o subprocesso termina, limpa estado e interface
        code = self.process.returncode if self.process else None
        self._log(f"{self.running_script} terminou (código {code}).")
        self.status_var.set("Pronto")
        self.btn_stop.config(state="disabled")
        self.process = None
        self.running_script = None
        self.start_time = None
        self.timer_var.set("00:00:00")

    # ─────────────── STOP ───────────────
    def _stop_process(self):
        if not self.process:
            return
        self._log("A terminar processo…")
        try:
            if sys.platform.startswith("win"):
                self.process.send_signal(signal.CTRL_BREAK_EVENT)  # envia CTRL+BREAK no Windows
            else:
                self.process.terminate()  # envia sinal SIGTERM no Unix
        except Exception as exc:
            self._log(f"Erro ao enviar sinal: {exc}")
        self.after(2000, self._force_kill_if_alive)  # espera 2s antes de forçar encerramento

    def _force_kill_if_alive(self):
        # Força kill se o processo ainda estiver ativo
        if self.process and self.process.poll() is None:
            self._log("Kill forçado do processo.")
            self.process.kill()

    # ─────────────── Timer ───────────────
    def _update_timer(self):
        if self.start_time is not None:
            elapsed = int(time.time() - self.start_time)
            hrs, rem = divmod(elapsed, 3600)
            mins, secs = divmod(rem, 60)
            self.timer_var.set(f"{hrs:02d}:{mins:02d}:{secs:02d}")
        self.after(1000, self._update_timer)  # atualiza a cada segundo


# ─────────────────── main ────────────────────
if __name__ == "__main__":
    app = UR3eGUI()     # instancia a GUI
    app.mainloop()      # inicia o loop da aplicação Tkinter
