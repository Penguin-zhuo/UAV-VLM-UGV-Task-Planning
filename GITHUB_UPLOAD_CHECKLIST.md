# GitHub upload checklist

Before making the repository public:

- [ ] Confirm that no API key, token, password, or `.env` file is present.
- [ ] Confirm whether the example UTM coordinates may be made public.
- [ ] Run `python -m py_compile` on the three Python scripts.
- [ ] Install `requirements.txt` in a clean Python 3.10 environment and run the example planner.
- [ ] Review `planner_config.json` and replace example vehicle parameters if necessary.
- [ ] Update the README citation section after the paper DOI is available.
- [ ] Create a GitHub Release such as `v1.0.0-paper`.
- [ ] Archive that release on Zenodo and add the Zenodo DOI to the README and manuscript Code Availability statement.
