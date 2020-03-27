.PHONY: docs test

docs:
	sphinx-apidoc -f -o docs tfplan
	rm -R docs/_build/html && sphinx-build docs docs/_build/html

test:
	pytest tests/*.py -sv --disable-warnings 2>/dev/null
