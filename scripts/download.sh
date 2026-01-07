#!/bin/bash

set -ex

aws s3 sync s3://soccernet-v2-amateur/ ./
