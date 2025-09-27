from lib.logger import get_logger
log = get_logger(__name__)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hex_decoder.py
Библиотека для декодирования HEX сообщений EPIRB/ELT маяков
Извлекает и декодирует поля из 144-битных сообщений COSPAS-SARSAT
"""

from typing import List, Iterable, Dict, Any

# ---- Вспомогательные функции для работы с битами ----

def hex_to_bits(h: str) -> List[int]:
    """
    Преобразует HEX строку в список битов (MSB-first)

    Args:
        h: HEX строка (может содержать пробелы и префикс 0x)

    Returns:
        Список битов (0 или 1)
    """
    h = ''.join(h.split()).lower()
    if h.startswith('0x'):
        h = h[2:]
    bits: List[int] = []
    for c in h:
        v = int(c, 16)
        bits.extend([(v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1])  # MSB-first nibble
    return bits

def bitslice(bits: List[int], start_1b: int, end_1b: int) -> List[int]:
    """
    Извлекает срез битов (нумерация с 1)

    Args:
        bits: Список битов
        start_1b: Начальный бит (нумерация с 1)
        end_1b: Конечный бит включительно (нумерация с 1)

    Returns:
        Срез битов, дополненный нулями если необходимо
    """
    s = start_1b - 1
    e = end_1b
    out = bits[s:min(e, len(bits))]
    if len(out) < (end_1b - start_1b + 1):
        out = out + [0] * ((end_1b - start_1b + 1) - len(out))
    return out

def bits_to_str(b: Iterable[int]) -> str:
    """Преобразует биты в строку '0' и '1'"""
    return ''.join('1' if int(x) else '0' for x in b)

def bits_to_int(b: Iterable[int]) -> int:
    """Преобразует биты в целое число"""
    v = 0
    for bit in b:
        v = (v << 1) | int(bit)
    return v

def bits_to_hex(b: Iterable[int]) -> str:
    """Преобразует биты в HEX строку с префиксом 0x"""
    bb = list(b)
    pad = (-len(bb)) % 4
    bb = [0]*pad + bb
    v = bits_to_int(bb)
    width = (len(bb)+3)//4
    return f"0x{v:0{width}X}"

# ---- Декодеры полей ----

def decode_format_flag(bit: int) -> str:
    """Декодирует флаг формата сообщения"""
    return "Long Format" if bit == 1 else "Short Format"

def decode_protocol_flag(bit: int) -> str:
    """Декодирует флаг протокола"""
    return "Standard/National/RLS" if bit == 0 else "User/Other"

def decode_protocol_code(n: int) -> str:
    """Декодирует код протокола"""
    protocols = {
        0x0: "Avionic",
        0x2: "ELT-DT",
        0x3: "Serial",
        0x6: "Radio Call Sign",
        0x8: "Orbitography",
        0xE: "SLP Test",
        0xF: "National Use"
    }
    return protocols.get(n, f"Protocol {n}")

def is_default_lat_pdf1(bits: List[int]) -> bool:
    """Проверяет является ли широта PDF-1 значением по умолчанию"""
    return bits_to_str(bits) == "0111111111"

def is_default_lon_pdf1(bits: List[int]) -> bool:
    """Проверяет является ли долгота PDF-1 значением по умолчанию"""
    return bits_to_str(bits) == "01111111111"

def is_default_latlon_pdf2(bits: List[int]) -> bool:
    """Проверяет является ли координата PDF-2 значением по умолчанию"""
    s = bits_to_str(bits)
    return s in {"0111111111", "0000000000", "1111111111"}

def decode_pos_source(bit: int) -> str:
    """Декодирует источник позиционных данных"""
    return "Internal" if bit == 1 else "External/Unknown"

def decode_1215(bit: int) -> str:
    """Декодирует наличие передатчика 121.5 МГц"""
    return "Included in beacon" if bit == 1 else "Not included"

def decode_country_code(code: int) -> str:
    """
    Декодирует код страны MID (Maritime Identification Digit)

    Args:
        code: Числовой код страны

    Returns:
        Название страны или код если неизвестен
    """
    # Некоторые основные коды стран
    countries = {
        111: "Chile",
        201: "Albania",
        202: "Andorra",
        203: "Austria",
        204: "Azores",
        205: "Belgium",
        206: "Belarus",
        207: "Bulgaria",
        208: "Vatican",
        209: "Cyprus",
        210: "Cyprus",
        211: "Germany",
        212: "Cyprus",
        213: "Georgia",
        214: "Moldova",
        215: "Malta",
        216: "Armenia",
        218: "Germany",
        219: "Denmark",
        220: "Denmark",
        224: "Spain",
        225: "Spain",
        226: "France",
        227: "France",
        228: "France",
        229: "Malta",
        230: "Finland",
        231: "Faroe Islands",
        232: "UK",
        233: "UK",
        234: "UK",
        235: "UK",
        236: "Gibraltar",
        237: "Greece",
        238: "Croatia",
        239: "Greece",
        240: "Greece",
        241: "Greece",
        242: "Morocco",
        243: "Hungary",
        244: "Netherlands",
        245: "Netherlands",
        246: "Netherlands",
        247: "Italy",
        248: "Malta",
        249: "Malta",
        250: "Ireland",
        251: "Iceland",
        252: "Liechtenstein",
        253: "Luxembourg",
        254: "Monaco",
        255: "Madeira",
        256: "Malta",
        257: "Norway",
        258: "Norway",
        259: "Norway",
        261: "Poland",
        262: "Montenegro",
        263: "Portugal",
        264: "Romania",
        265: "Sweden",
        266: "Sweden",
        267: "Slovak Republic",
        268: "San Marino",
        269: "Switzerland",
        270: "Czech Republic",
        271: "Turkey",
        272: "Ukraine",
        273: "Russia",
        274: "Macedonia",
        275: "Latvia",
        276: "Estonia",
        277: "Lithuania",
        278: "Slovenia",
        279: "Serbia",
        301: "Anguilla",
        303: "Alaska (USA)",
        304: "Antigua and Barbuda",
        305: "Antigua and Barbuda",
        306: "Curacao",
        307: "Aruba",
        308: "Bahamas",
        309: "Bahamas",
        310: "Bermuda",
        311: "Bahamas",
        312: "Belize",
        314: "Barbados",
        316: "Canada",
        319: "Cayman Islands",
        321: "Costa Rica",
        323: "Cuba",
        325: "Dominica",
        327: "Dominican Republic",
        329: "Guadeloupe",
        330: "Grenada",
        331: "Greenland",
        332: "Guatemala",
        334: "Honduras",
        336: "Haiti",
        338: "USA",
        339: "Jamaica",
        341: "St Kitts and Nevis",
        343: "St Lucia",
        345: "Mexico",
        347: "Martinique",
        348: "Montserrat",
        350: "Nicaragua",
        351: "Panama",
        352: "Panama",
        353: "Panama",
        354: "Panama",
        355: "Panama",
        356: "Panama",
        357: "Panama",
        358: "Puerto Rico",
        359: "El Salvador",
        361: "St Pierre and Miquelon",
        362: "Trinidad and Tobago",
        364: "Turks and Caicos Islands",
        366: "USA",
        367: "USA",
        368: "USA",
        369: "USA",
        370: "Panama",
        371: "Panama",
        372: "Panama",
        373: "Panama",
        374: "Panama",
        375: "St Vincent",
        376: "St Vincent",
        377: "St Vincent",
        378: "British Virgin Islands",
        379: "US Virgin Islands",
        516: "Russia",
    }
    return countries.get(code, f"Country {code}")

# ---- Структуры полей сообщения ----

def get_message_fields() -> List[tuple]:
    """
    Возвращает определение полей сообщения EPIRB

    Returns:
        Список кортежей (начальный_бит, конечный_бит, название_поля)
    """
    return [
        (1, 15,  "Bit-synchronization pattern"),
        (16, 24, "Frame-synchronization pattern"),
        (25, 25, "Format Flag"),
        (26, 26, "Protocol Flag"),
        (27, 36, "Country Code"),
        (37, 40, "Protocol Code"),
        (41, 64, "Test Data"),
        (65, 74, "Latitude (PDF-1)"),
        (75, 85, "Longitude (PDF-1)"),
        (86, 106,"BCH PDF-1"),
        (107,110,"Fixed (1101)"),
        (111,111,"Position data source"),
        (112,112,"121.5 MHz Homing Device"),
        (113,122,"Latitude (PDF-2)"),
        (123,132,"Longitude (PDF-2)"),
        (133,144,"BCH PDF-2"),
    ]

def decode_message(hex_message: str) -> Dict[str, Any]:
    """
    Полностью декодирует HEX сообщение EPIRB

    Args:
        hex_message: HEX строка сообщения (36 символов для длинного формата)

    Returns:
        Словарь с декодированными полями
    """
    bits = hex_to_bits(hex_message)
    if len(bits) != 144:
        # Дополняем или обрезаем до 144 бит
        bits = (bits + [0]*144)[:144]

    decoded = {
        "hex": hex_message,
        "bits": bits,
        "fields": []
    }

    for a, b, name in get_message_fields():
        bl = bitslice(bits, a, b)
        field_data = {
            "range": f"{a}-{b}" if a != b else f"{a}",
            "bits": bits_to_str(bl),
            "name": name,
            "decoded": decode_field(name, bl)
        }
        decoded["fields"].append(field_data)

    return decoded

def decode_field(name: str, bits: List[int]) -> str:
    """
    Декодирует отдельное поле по его названию

    Args:
        name: Название поля
        bits: Биты поля

    Returns:
        Декодированное значение в виде строки
    """
    if name == "Bit-synchronization pattern":
        return "Valid" if bits_to_str(bits) == "1"*15 else "Invalid"
    elif name == "Frame-synchronization pattern":
        pat = bits_to_str(bits)
        if pat == "000101111":
            return "Normal Operation"
        elif pat == "001011111":
            return "Self-Test"
        elif pat == "001001111":
            return "Test Protocol"
        else:
            return f"Unknown ({pat})"
    elif name == "Format Flag":
        return decode_format_flag(bits[0])
    elif name == "Protocol Flag":
        return decode_protocol_flag(bits[0])
    elif name == "Country Code":
        code = bits_to_int(bits)
        return f"{code} - {decode_country_code(code)}"
    elif name == "Protocol Code":
        return decode_protocol_code(bits_to_int(bits))
    elif name == "Test Data":
        return bits_to_hex(bits)
    elif name == "Latitude (PDF-1)":
        return "Default value" if is_default_lat_pdf1(bits) else f"bin {bits_to_str(bits)}"
    elif name == "Longitude (PDF-1)":
        return "Default value" if is_default_lon_pdf1(bits) else f"bin {bits_to_str(bits)}"
    elif name == "BCH PDF-1":
        return bits_to_hex(bits)
    elif name == "Fixed (1101)":
        return "Valid" if bits_to_str(bits) == "1101" else f"Invalid ({bits_to_str(bits)})"
    elif name == "Position data source":
        return decode_pos_source(bits[0])
    elif name == "121.5 MHz Homing Device":
        return decode_1215(bits[0])
    elif name in {"Latitude (PDF-2)", "Longitude (PDF-2)"}:
        return "Default value" if is_default_latlon_pdf2(bits) else f"bin {bits_to_str(bits)}"
    elif name == "BCH PDF-2":
        return bits_to_hex(bits)
    else:
        return bits_to_str(bits)

def build_table_rows(bits: List[int]) -> List[List[str]]:
    """
    Строит строки таблицы для отображения декодированных полей

    Args:
        bits: Список битов сообщения (144 бита)

    Returns:
        Список строк таблицы [диапазон, биты, название, декодированное_значение]
    """
    table = []
    for a, b, name in get_message_fields():
        bl = bitslice(bits, a, b)
        decoded = decode_field(name, bl)
        table.append([
            f"{a}-{b}" if a != b else f"{a}",
            bits_to_str(bl),
            name,
            decoded
        ])
    return table