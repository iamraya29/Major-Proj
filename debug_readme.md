
Your `social_nav_sim.py` was found because `find_packages()` `in setup.py` recursively included the `test_pkg.social_gym` package itself.
However — subfolders that don’t have their own `__init__.py` (like your custom_config/) do not get recognized as Python subpackages.
Just add an empty __init__.py file there