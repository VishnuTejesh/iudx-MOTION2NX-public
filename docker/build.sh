#!/bin/bash

# To be executed from project root
docker buildx build --cache-from type=registry,ref=ghcr.io/datakaveri/motion2nx:build  -t iudx/motion2nx:latest -f docker/Dockerfile .