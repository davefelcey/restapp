# restapp

## Overview
A REST service to identify requirements in a job specification

## Setup
To setup for development;
* pipenv install
* ...
* pipenv lock

To deploy on Heroku;
1. heroku login
2. git clone https://github.com/davefelcey/restapp.git
3. cd restapp
4. heroku create
5. heroku buildpacks:add heroku/jvm
6. git push heroku master
7. heroku ps:scale web=1
8. heroku open

## Test
Execute the command;
```
curl -G -v "http://<url of service>/" --data-urlencode "url=<job description url"
```
The response will be a list of questions to ask. For example;
```
Do you have coding experience using Java and Android?
Do you have programming experience with Java essential?
```

