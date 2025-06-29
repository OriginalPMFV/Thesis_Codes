import sys
import logging
import time
import itertools

import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

from rotations import rotations  # lista de 121 rotações (rx, ry, rz)

# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def principal(deteccoes):
    """Recebe uma lista de detecções na forma:
        [(modo, x, y, z, largura), ...]
    e gerencia todo o ciclo de comunicação RTDE com fallback automático de
    rotações caso a solução cinemática não seja encontrada.
    """
    test_robot_communication(deteccoes)


# ─────────────────────────────────────────────────────────────────────────────
# ROTINA DE COMUNICAÇÃO COM O ROBÔ (RTDE)
# ─────────────────────────────────────────────────────────────────────────────

def test_robot_communication(cor_list):
    """Gerencia o diálogo completo via RTDE, tentando múltiplas rotações por
    detecção até encontrar a primeira que satisfaça
    ``output_int_register_2 == 1``.

    Parameters
    ----------
    cor_list : list[tuple]
        Cada elemento deve conter (modo, x, y, z, largura).
    """

    # ░░░ Configuração de rede / RTDE ░░░
    ROBOT_HOST = "192.168.250.70"
    ROBOT_PORT = 30004
    CONFIG_XML = "control_loop_configuration.xml"
    FREQUENCY = 500  # Hz

    logging.getLogger().setLevel(logging.INFO)

    # Carrega recipes do ficheiro XML (descrição dos pacotes RTDE)
    conf = rtde_config.ConfigFile(CONFIG_XML)
    state_names, state_types = conf.get_recipe("state")         # pacotes de leitura
    setp_names, setp_types = conf.get_recipe("setp")            # pacotes de escrita
    watchdog_names, watchdog_types = conf.get_recipe("watchdog")

    # Conecta ao robô (loop até estabelecer ligação)
    con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
    while con.connect() != 0:
        time.sleep(0.5)
    print("--------------- Conectado ao robô -------------\n")
    con.get_controller_version()

    # Prepara canais de entrada/saída conforme receitas
    con.send_output_setup(state_names, state_types, FREQUENCY)
    setp = con.send_input_setup(setp_names, setp_types)
    watchdog = con.send_input_setup(watchdog_names, watchdog_types)

    # Inicializa valores dos registradores
    for i in range(7):
        setattr(setp, f"input_double_register_{i}", 0.0)

    watchdog.input_bit_registers0_to_31 = 0
    setp.input_int_register_0 = 99
    setp.input_int_register_1 = 0
    setp.input_int_register_2 = 0

    # Inicia sincronização RTDE com o robô
    if not con.send_start():
        sys.exit("Erro ao iniciar sincronização RTDE com o robô")
    con.send(setp)
    con.send(watchdog)

    # Espera sinal do robô indicando que está pronto
    print("Verificando estado inicial do robô…")
    while True:
        state = con.receive()
        if state and getattr(state, "output_int_register_1", 0) == 1:
            print("Robô pronto para receber coordenadas (output_int_register_1 == 1)")
            break
        time.sleep(0.5)

    # ────────────────────────────────────────────────────────────────────────
    #  Laço principal: processa cada detecção tentando rotações em sequência
    # ────────────────────────────────────────────────────────────────────────

    for idx_det, (modo, x, y, z, largura) in enumerate(cor_list, start=1):
        print(f"\n▶▶ Detecção {idx_det}: modo={modo}, x={x:.3f}, y={y:.3f}, z={z:.3f}, largura={largura:.3f}")

        rot_iter = iter(rotations)        # iterador sobre rotações pré-definidas
        solucionado = False
        tentativa = 0

        while not solucionado:
            try:
                rx, ry, rz = next(rot_iter)   # tenta próxima rotação
            except StopIteration:
                print(f"⚠ Detecção {idx_det} ignorada: nenhuma rotação levou a solução cinemática.")
                break  # sai do while-not-solucionado → passa para a próxima detecção

            tentativa += 1
            print(f"  • Tentativa {tentativa} usando rotação (rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f})…")

            # Preenche registradores com dados da posição e orientação
            setp.input_double_register_0 = x
            setp.input_double_register_1 = y
            setp.input_double_register_2 = z
            setp.input_double_register_3 = rx
            setp.input_double_register_4 = ry
            setp.input_double_register_5 = rz
            setp.input_double_register_6 = largura * 1000 - 20  # compensação específica do projeto
            setp.input_int_register_0 = modo

            # Envia os dados ao robô
            con.send(setp)
            con.send(watchdog)
            print("    ⮕ Setpoint enviado.")

            # Espera resposta indicando se a solução cinemática é válida
            if "output_int_register_2" in state_names:
                while True:
                    state = con.receive()
                    if not state or not hasattr(state, "output_int_register_2"):
                        print("    ⚠ Sem resposta de output_int_register_2, tentando novamente…")
                        time.sleep(0.05)
                        continue

                    setp.input_int_register_2 = 0  # reset de flag

                    if state.output_int_register_2 == 1:
                        print("    ✔ Solução cinemática encontrada!")
                        solucionado = True
                        time.sleep(0.5)
                        break
                    elif state.output_int_register_2 == 2:
                        print("    ✖ Sem solução para esta orientação (output_int_register_2 == 2). Próxima rotação…")
                        time.sleep(0.5)
                        break
                    else:
                        time.sleep(0.5)
            else:
                # Se campo não existir na receita, assume-se sucesso
                print("    ⚠ output_int_register_2 não está na recipe; assumindo solução.")
                solucionado = True

        # Espera início do movimento: output bit0 == 1
        print("    Aguardando bit0==1 (start Polyscope)…")
        while True:
            state = con.receive()
            if state and (getattr(state, "output_bit_registers0_to_31", 0) & 0x1):
                print("    ▶ Movimento iniciou (bit0==1)")
                break
            time.sleep(0.05)

        # Espera fim do movimento: bit0 volta a 0
        print("    Aguardando bit0==0 (movimento concluído)…")
        while True:
            state = con.receive()
            if state and (getattr(state, "output_bit_registers0_to_31", 1) & 0x1) == 0:
                print("    ✔ Movimento concluído\n")
                break
            time.sleep(0.05)

    # ────────────────────────────────────────────────────────────────────────
    # Encerramento
    # ────────────────────────────────────────────────────────────────────────

    print("Todas as detecções processadas. Enviando modo 3 para encerrar o ciclo RTDE…")
    setp.input_int_register_0 = 99
    con.send(setp)
    time.sleep(0.05)
    con.receive()
    con.disconnect()

    try:
        con.send_pause()
    finally:
        con.disconnect()
    print("Ciclo RTDE finalizado com sucesso.")


# ─────────────────────────────────────────────────────────────────────────────
# Entrada por linha de comando (exemplo rápido de teste)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Exemplo: python control_script.py 2 0.01 0.058 0.406 0.072
    if len(sys.argv) == 6:
        modo, x, y, z, largura = map(float, sys.argv[1:])
        deteccoes = [(int(modo), x, y, z, largura)]
        principal(deteccoes)
    else:
        print("Uso: python control_script.py <modo> <x> <y> <z> <largura>")
