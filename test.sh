#!/bin/bash

# URL='https://www.indeed.co.uk/cmp/Educ8-Coventry-&-Warwickshire-Ltd/jobs/Learning-Support-Assitant-fe6ae3f3c882c2b0?sjdu=QwrRXKrqZ3CNX5W-O9jEvTtCYQ0sua_Xry5GQQ8NuaRcC-_SHkf5ShOUs-V19F3zbFmkCUkK9d69ahkt-HB715eYoTEBxP-4wkdfbenCMc8&tk=1c99uoh649tok9uq&vjs=3'
APP=https://qrestapp.herokuapp.com/
# APP=http://127.0.0.1:8000

curl -v -G -v "$APP" --data-urlencode "url=$1"
