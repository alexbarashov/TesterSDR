#!/usr/bin/env python3
# epirb_hex_decoder.py
# Использует библиотеку hex_decoder для декодирования EPIRB сообщений
# Отображает результаты в окне Matplotlib

import sys
import os
import argparse
import matplotlib.pyplot as plt

# Добавляем путь к библиотекам
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lib.hex_decoder import hex_to_bits, build_table_rows

# Примеры сообщений для тестирования
DEFAULT_HEX = "FFFED080020000007FDFFB0020B783E0F66C"
HEX_EXAMPLES = {
    "default": "FFFED080020000007FDFFB0020B783E0F66C",
    "test1": "FFFED08C9EAF0F0F7FDFFFD230B783E0F66C",
    "test2": "FFFE2F8C9EAF0F0F7FDFFFD230B783E0F66C",
    "test3": "FFFED080020000007FDFFB0020B783E0F66C"
}

def show_epirb_params_window(hex_message: str):
    """
    Отображает декодированные параметры EPIRB в окне Matplotlib

    Args:
        hex_message: HEX строка сообщения EPIRB (144 бита / 36 hex символов)
    """
    # Преобразуем HEX в биты
    bits = hex_to_bits(hex_message)
    if len(bits) != 144:
        # Дополняем или обрезаем до 144 бит
        bits = (bits + [0]*144)[:144]

    # Заголовки таблицы
    headers = ["Binary Range", "Binary Content", "Field Name", "Decoded Value"]

    # Получаем данные для таблицы
    rows = build_table_rows(bits)

    # Создаем окно Matplotlib с таблицей
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Заголовок окна
    fig.suptitle(f"EPIRB/ELT Beacon Parameters Decoder\nHEX: {hex_message}", fontsize=11, fontweight='bold')

    # Создаем таблицу
    tbl = ax.table(cellText=[headers] + rows,
                   loc='center',
                   cellLoc='left',
                   colWidths=[0.12, 0.25, 0.28, 0.35])

    # Настройка стилей таблицы
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)

    # Стиль заголовка
    for i in range(len(headers)):
        cell = tbl[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
        cell.set_height(0.05)

    # Чередующиеся цвета строк
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            cell = tbl[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            cell.set_height(0.04)

    # Настройка окна
    try:
        fig.canvas.manager.set_window_title("EPIRB/ELT Beacon Decoder")
    except Exception:
        pass

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Декодер EPIRB/ELT beacon сообщений COSPAS-SARSAT",
        epilog="Примеры использования:\n"
               "  %(prog)s                                      # использует сообщение по умолчанию\n"
               "  %(prog)s FFFED080020000007FDFFB0020B783E0F66C  # декодирует указанное сообщение\n"
               "  %(prog)s --example test1                      # использует встроенный пример",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("hex",
                       nargs="?",
                       default=DEFAULT_HEX,
                       help="HEX сообщение (36 hex символов для длинного формата, 144 бита)")

    parser.add_argument("--example",
                       choices=list(HEX_EXAMPLES.keys()),
                       help="Использовать встроенный пример сообщения")

    args = parser.parse_args()

    # Выбираем сообщение для декодирования
    if args.example:
        hex_message = HEX_EXAMPLES[args.example]
        print(f"Используется пример '{args.example}': {hex_message}")
    else:
        hex_message = args.hex
        if hex_message == DEFAULT_HEX:
            print(f"Используется сообщение по умолчанию: {hex_message}")

    # Отображаем окно с декодированными параметрами
    show_epirb_params_window(hex_message)

if __name__ == "__main__":
    main()
