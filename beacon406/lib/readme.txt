Коммит-месседж:

3.demodмс
получает массив фазы
находит фронты 
и вычисляет параметры 

нужно обьединить с metrics чтобы считать метки фронтов и непрерывную фазу за один проход... 



2. metrics
metrics.py готовит фазу из IQ.

оптимизация быстродейсвия - 
xs_ms - не нужна надо избавится от времени во всех таких обработках
на входе самлы на выходе семплы надо будет только учесть частоту децимации. 
время вычисляется только там где реально надо ...UI данные и графики 

надежность - 
использовать FSK для надежной фазы ... 

1.beckends.py

plot/backends: file one-shot + safe EOF; SDR underflow no-stop
- FilePlaybackBackend: single-pass playback, inverse IF applied only for file, stop() resets
- Plot: stop only when BACKEND_NAME=="file"; SDR empty blocks are skipped

тестирование все железо работает авто и файл тоже работает 
SDR mode :
+RTL
+HackRF
+Airspy
+SDRplay - the best of the best !!!
+File
+Textronix
+auto
