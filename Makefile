# Makefile for Audio Chronicle Accelerator tests

.PHONY: test test-e2e test-all

# Run unit tests (skip E2E — ACCELERATOR_URL not set)
test:
	pytest tests/ -v --ignore=tests/e2e

# Run E2E tests against a live local server
test-e2e:
	ACCELERATOR_URL=http://localhost:8765 pytest tests/e2e/ -v

# Run all tests (E2E skipped automatically if ACCELERATOR_URL not set)
test-all:
	pytest tests/ -v
