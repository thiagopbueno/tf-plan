.PHONY: docs test upload

docs:
	sphinx-apidoc -f -o docs tfplan
	[ -e "docs/_build/html" ] && rm -R docs/_build/html
	sphinx-build docs docs/_build/html

test:
	pytest tests/*.py -sv --disable-warnings 2>/dev/null

upload:
	[ -e "dist/" ] && rm -Rf dist/
	python3 setup.py sdist bdist_wheel
	twine upload dist/*
