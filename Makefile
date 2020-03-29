.PHONY: docs test upload

docs:
	sphinx-apidoc -f -o docs tfplan
	rm -R docs/_build/html && sphinx-build docs docs/_build/html

test:
	pytest tests/*.py -sv --disable-warnings 2>/dev/null

upload:
	python3 setup.py clean sdist
	twine upload dist/*
