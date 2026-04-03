# Zestaw narzędzi do treningu agentów opartych na algorytmach RL dla platformy RAG-2

Repozytorium zawiera zestaw skryptów służących do trenowania i testowania modeli sztucznej inteligencji opartych na algorytmach uczenia ze wzmocnieniem (Reinforcement Learning). Projekt zaimplementowany został w języku Python 3.9.13. Rozwiązanie zostało oparte na projekcie <https://github.com/DLR-RM/rl-baselines3-zoo>.

Repozytorium jest forkiem projektu <https://github.com/theImmortalCoders/rag-2-ai-training_tool>.

Autor: [@bkrowka](https://github.com/bkrowka)

## Instalacja i konfiguracja

1. **Utworzenie i aktywacja wirtualnego środowiska**
   - **Linux / macOS:**
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     venv\Scripts\Activate
     ```

2. **Zmiana wersji pip, setuptools i wheel**

   ```bash
   python -m pip install --upgrade pip==21.1.2 setuptools==66 wheel==0.38.0
   ```

3. **Instalacja wymaganych pakietów:**
   ```bash
   pip install -r requirements.txt
   ```

## Rozwiązywanie problemów

W repozytorium znajduje się plik `full_requirements.txt`, który zawiera pełną listę pakietów wraz z wersjami, na których projekt był testowany i może służyć jako punkt odniesienia podczas ręcznej instalacji zależności lub diagnozowania problemów. W przypadku wystąpienia błędów podczas instalacji pakietów, zaleca się wyszukiwanie gotowych plików binarnych w formacie `.whl` w ogólnodostępnych repozytoriach internetowych, na przykład na stronie www.piwheels.org. Problematyczne pakiety mogą być wówczas instalowane ręcznie z wykorzystaniem pobranych plików. Alternatywnie możliwe jest samodzielne budowanie plików `.whl` dla wybranych bibliotek.

**Instalacja zależność z lokalnego katalogu:**

```bash
pip install --no-index --find-links=packages nazwa_pakietu
```

## Podstawowa obsługa

Trening nowego agenta:

```
python train.py --env "env_id" --algo "algo_name"
```

Ewaluacja ostatnio wytrenowanego agenta:

```
python evaluate.py --env "env_id" --algo "algo_name" --deterministic
```

Uruchomienie ostatnio wytrenowanego agenta w trybie ciągłym:

```
python enjoy.py --env "env_id" --algo "algo_name" --deterministic
```
