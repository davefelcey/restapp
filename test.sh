#!/bin/bash

# URL='https://www.indeed.co.uk/viewjob?jk=b39ccb985a9de5e1&from=tp-serp&tk=1c29ob0im14he6rl'
APP=https://qrestapp.herokuapp.com/
# APP=http://127.0.0.1:8000

curl -v -G -v "$APP" --data-urlencode "url=$1"
