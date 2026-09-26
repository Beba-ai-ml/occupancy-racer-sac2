# Trening E - pierwsza godzina bez wywołań agenta

Mapy E_02-E_07 są treningowe; E_08-E_10 pozostają poza uczeniem. Przy starcie
zapisz kopie PGM i zones PNG oraz pełne konfiguracje w katalogu sesji. Render
wyłączony. Nowy profil `game_small.yaml` ma `distance_progress.mode: free`:
nagroda za postęp dotyczy ruchu do przodu auta, bez kierunku wokół środka mapy.
Nie oznacza to planowania globalnej trasy ani gwarancji unikania jazdy w kółko.

Plan sprzętowy z pomiarów 26.09: RTX3090 i RTX3080, po jednej niezależnej próbie
SAC na każdej, po 8 actorów CPU. Jedna karta nie jest drugim learnerem tego
samego modelu. Używaj GPU UUID w CUDA_VISIBLE_DEVICES: numery PyTorch i
nvidia-smi na tej maszynie są w różnej kolejności. Jeden wątek BLAS/PyTorch
na actor zapobiega nadmiernej liczbie wątków. Bufor 750000 na próbę oszczędza
RAM przy dwóch równoczesnych sesjach; batch256, UTD2, [512,512,256], gamma0.99,
alpha_min0.05 pochodzą z wcześniejszych udanych treningów 450-ray.

Trener zapisuje `health.json` co około 30 sekund oraz `map_path`, `transitions`
i `updates` w CSV epizodów. Pętla wykrywa utratę actora i niefinitywne straty.
SIGTERM/SIGINT zatrzymuje główną pętlę, zapisuje bieżący checkpoint po pełnym
kroku uczenia i sprawdzeniu skończoności wag/optimizera, oraz zamyka własne
procesy potomne. Wyjątek w uczeniu zachowuje poprzedni checkpoint.
`--seed` ustala learner, replay i osobne ziarna actorów; asynchroniczna kolejność
doświadczeń nadal nie jest deterministyczna.
`--resume-warmup-steps 5000` pozwala po awarii zebrać świeży bufor bez starego
rozgrzewania skalowanego liczbą actorów. Bez tej opcji starszy tryb jest zachowany.

Usługi systemd mają utrzymywać trening niezależnie od terminala i sesji agenta,
z ograniczeniami RAM i liczbą ponowień. Monitor uruchamia się osobno:

```bash
python tools/monitor_training.py \
  --session mapper-e-a.service /absolute/path/to/session_A \
  --session mapper-e-b.service /absolute/path/to/session_B \
  --duration 3600 --interval 60 --output /absolute/path/to/monitor
```

Monitor zapisuje kontrole do `checks.jsonl`: procesy, świeżość heartbeat,
postęp kroków i aktualizacji, wartości strat, wolny RAM i dysk. Nie ocenia
dystansu, liczby zer ani wczesnej jakości modelu. W razie awarii wykonuje
najwyżej jeden restart każdej wskazanej usługi i zapisuje wynik. Niska jakość
w pierwszej godzinie nie powoduje restartu. Po godzinie zapisuje
`finished.json` i kończy proces, pozostawiając trening. Nie wywołuje LLM,
nie budzi agenta, nie wysyła wiadomości. Nie udaje ręcznej obserwacji przez godzinę.

Pełny test uruchomienia sprawdza prawdziwy proces `src.train_ssac`, plik CSV,
co najmniej jedną aktualizację GPU i końcowy checkpoint na syntetycznie krótkich
epizodach. Testy fizyki i monitora: `python -m unittest discover -s tests -v`.
