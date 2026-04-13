# option-pricer
Option Pricer

**Overview**
- **Option Pricer** is a small Python library and Streamlit demo for pricing a variety of option products using both closed-form and simulation methods.

**Files**
- **app.py**: Streamlit application providing a web UI to price products interactively. Launch with `streamlit run app.py`.
- **option_pricer.py**: Core implementation containing the `OptionPricer` class. Supported methods include:
	- `get_european_option_price()` — Black–Scholes closed-form pricing (call/put).
	- `get_american_option_price()` — Binomial tree (American option, early exercise).
	- `get_geometric_asian_option_price()` — Closed-form geometric Asian price.
	- `get_arithmetic_asian_option_price()` — Monte Carlo arithmetic Asian with control variate.
	- `get_geometric_basket_option_price()` — Closed-form geometric basket (two assets).
	- `get_arithmetic_basket_option_price()` — Monte Carlo arithmetic basket with optional control variate.
	- `get_KIKO_put_option_price()` — KIKO (knock-in knock-out) put via Sobol quasi-MC sampling.
- **implied_vol.py**: `get_implied_volatility()` helper to find Black–Scholes implied volatility via Newton-Raphson.

**Installation**
- Recommended Python: 3.8+
- Install required packages:

```bash
pip install numpy scipy pandas streamlit
```

If you use a virtual environment, activate it first (venv, conda, etc.).

**Run the Streamlit demo**

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit to interact with the UI.

**Basic usage (python)**
- Example: price a European call from a script or REPL

```python
from option_pricer import OptionPricer

pricer = OptionPricer(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)
price = pricer.get_european_option_price(option_type='call')
print('European call price:', price)
```

**Notes & Behavior**
- Several Monte Carlo routines use a fixed seed (`42`) for reproducible runs.
- The arithmetic Asian and basket pricers include an optional control variate implementation to reduce variance.
- The American pricer uses a binomial tree with `n_steps` (adjustable in the UI).

**Contributing / Extending**
- Add unit tests or additional products by editing `option_pricer.py` and exposing UI controls in `app.py`.
- Consider adding a `requirements.txt` or a `pyproject.toml` for reproducible installs.

**License**
- No license file is included. Add one if you intend to publish or share this code.

**Reference**
- See [app.py](app.py), [option_pricer.py](option_pricer.py), and [implied_vol.py](implied_vol.py) for implementation details.
