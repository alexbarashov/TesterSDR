# config_backend.py
# Конфиг выбора SDR без правки основного кода/GUI.
# Стиль путей по умолчанию: raw-строки + прямые слэши, как мы зафиксировали.
# "auto" | "soapy_hackrf" | "soapy_airspy" | "soapy_sdrplay" | "file"
#BACKEND_NAME = "soapy_rtl"
#BACKEND_NAME = "soapy_hackrf"
#BACKEND_NAME = "soapy_airspy"
#BACKEND_NAME = "soapy_sdrplay"  # the best !
#BACKEND_NAME = "rsa306"
BACKEND_NAME = "auto"
BACKEND_ARGS = None  # можно оставить None, SoapySDR сам найдёт SDR

#BACKEND_NAME = "file"
#BACKEND_ARGS = r"C:/work/TesterSDR/captures/psk406msg_f150.cf32"
