# Fizyka małych map - 26.09.2026

Profil `config/config_sac_small.yaml` służy do treningów małych map.
Checkpoint zachowuje wagi, ale wznowiony trening używa pliku fizyki wskazanego
przez aktualną konfigurację. Stare sesje zachowują poprzednią fizykę tylko wtedy,
gdy wskazują jej niezmienioną kopię. Profil dużej hali nie został jeszcze ustalony.
Zmiana nie dotyka repozytorium ROS ani pojazdu.

## Sterowanie i limity

Wartości nominalne ustalone z użytkownikiem:

| Manewr | Zwłoka od polecenia | Ruch kół | Całość |
|---|---:|---:|---:|
| Środek do pełnego skrętu | 150 ms | 150 ms | 300 ms |
| Pełny skręt do przeciwnego pełnego skrętu | 150 ms | 300 ms | 450 ms |

Czas liczy zegar symulacji. Każda zmiana polecenia dociera po zwłoce, także
gdy sieć zdążyła wydać kolejną komendę. Koła poruszają się z ograniczoną
szybkością, bez dodatkowego starego filtra serwa. Pozostaje dotychczasowy
filtr narastania prędkości obrotu nadwozia (yaw); czasy z tabeli dotyczą kół.
Reset epizodu czyści całą kolejkę i stan serwa.

Na każdy epizod niezależnie losowane są mnożniki czasu przesyłania i czasu
przestawienia z zakresu 0,9-1,1. To robocza propozycja zmienności, nie pomiar
rozkładu opóźnień sprzętu. Aby uzyskać dokładnie nominalne czasy, ustaw
`timing_scale_range: [1.0, 1.0]` w `config/physics_small.yaml`.

Na początku epizodu wybierany jest z jednakowym prawdopodobieństwem fizyczny
limit 1 / 1,5 / 2 m/s. Sieć nadal wydaje polecenia przyspieszania z dotychczasowego
zakresu 0-2 m/s²; limit ogranicza wykonany ruch. Pomiar prędkości to rzeczywista
prędkość podzielona przez stałe 2,5 m/s, zgodnie z ROS. Osiągnięcie limitu 1 m/s
daje wejście 0,4, a nie 1,0. Nowy profil wyłącza szum, dryf i opóźnienie tylko
kanału prędkości, aby to zachować. Nagroda za prędkość odnosi się do nominalnego
maksimum 2 m/s, nie do losowanego ograniczenia. Limity nie dopisują nowego wejścia.

## Zakręty

27.09.2026 użytkownik skorygował średnice przy pełnym skręcie: 2,5 m przy
1,5 m/s i 3 m przy 2 m/s. Punkt 1,5 m przy wolnej jeździe pozostał. Profil
stosuje efektywną krzywą poszerzania zakrętu:

`średnica(v) = 1,5 + (D2 - 1,5) * (|v| / 2)^p`, gdzie `D2` jest
losowane z zakresu 2-3 m, a
`p = log((2,5 - 1,5) / (3 - 1,5)) / log(1,5 / 2)`.

Od 27.09.2026 trening losuje w każdym epizodzie średnicę przy 2 m/s
jednostajnie z zakresu 2-3 m. Średnica przy 1,5 m/s skaluje się razem z nią
od około 1,83 do 2,5 m, aby krzywa pozostała spójna. Czas reakcji serwa nie
zmienia się przez to losowanie.

Jest to przybliżenie z trzech oszacowań, nie pełny model sił opony i bocznego
poślizgu. Punkt „wolno” nie ma zmierzonej prędkości, dlatego 1,5 m jest granicą
przy prędkości bliskiej zeru. Dla górnego wariantu `D2=3` przy 1 m/s
średnica wynosi około 2,065 m, przy 1,5 m/s 2,5 m, a przy 2 m/s 3 m.
Częściowy skręt daje mniejszą krzywiznę; znaki
skrętu pozostają bez zmian. Ten model zastępuje starą redukcję kąta i poślizg
aktywowany dopiero powyżej 4 m/s, aby nie naliczać obu naraz.

## Zgodność z autem i odtwarzanie

- Bez zmian kątów, kolejności promieni i architektury: 450 promieni + 5 kanałów,
  4 klatki, 1820 wejść. Kanał serwa nadal opisuje komendę, nie ukryty stan fizyki.
- ROS nadal używa offset=-90, direction=-1, TF yaw=0 i dzielnika prędkości 2,5.
- Zapis CSV `config_json` zawiera teraz także pełne `physics_cfg`, oprócz
  `sim_cfg` i argumentów. Odtwarzanie powinno korzystać z tego zapisu.
- Render spowalnia wyświetlanie, lecz w nowym profilu nie zmienia kroku fizyki.
- Ponieważ nie rozszerzamy obserwacji o kolejkę komend, wynik uczenia z nową
  zwłoką trzeba sprawdzić empirycznie; same testy fizyki nie dowodzą skutecznej jazdy.

## Użycie po przygotowaniu map

`J_05` w konfiguracji to istniejąca mapa do przygotowania środowiska. Użytkownik
przebudowuje mapy; ich finalnej listy treningowej i testowej jeszcze nie ustalono.
Przed treningiem wskaż właściwą mapę/pulę i osobny identyfikator sesji. Przykład:

```bash
python -m src.train_ssac --config-file config/config_sac_small.yaml \
  --map J_05 --session-id J_SMALL_20260926_01 --no-resume
```

`--render` wyświetla trening. Samo `--no-resume` nie tworzy nowego katalogu,
dlatego za każdym eksperymentem zmień `--session-id`. Przeciwnik i dodatkowe
losowane przeszkody są wyłączone w tym profilu; kartony mają pochodzić z map.
Wymagane zależności, w tym Pillow do stref PNG, są w `requirements.txt`.
Stare `run.py` nie jest podglądem polityki 1820-wejściowej.

Testy fizyki i środowiska, bez treningu i bez ROS:

```bash
python -m unittest discover -s tests -v
```
