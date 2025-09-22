Коммит-месседж:

1.beckends.py

plot/backends: file one-shot + safe EOF; SDR underflow no-stop
- FilePlaybackBackend: single-pass playback, inverse IF applied only for file, stop() resets
- Plot: stop only when BACKEND_NAME=="file"; SDR empty blocks are skipped

SDR mode :
+RTL
+HackRF
+Airspy
+SDRplay - the best of the best !!!
+File
+Textronix
+auto
