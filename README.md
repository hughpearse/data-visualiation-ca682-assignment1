# Visualising Summary Statistics for London Boroughs

## Requirements
To run this you will need Python v2.7 with Tkinter

## Installation Instructions:
```bash
foo@bar:~$ sudo pip install virtualenv #install python sandboxing to the OS
foo@bar:~$ git clone https://github.com/hughpearse/data-visualiation-ca682-assignment1
foo@bar:~$ cd data-visualiation-ca682-assignment1/src
foo@bar:~$ virtualenv sandbox #create python sandbox
foo@bar:~$ virtualenv -p /usr/bin/python2.7 sandbox #add python 2.7 to sandbox
foo@bar:~$ source sandbox/bin/activate #enter sandbox
foo@bar:~$ pip install numpy pandas matplotlib xlrd scipy #install project deps
foo@bar:~$ chmod 700 ./A02-program.py #change file permissions
foo@bar:~$ deactivate #exit sandbox
```

## Usage Instructions:
```bash
foo@bar:~$ cd src
foo@bar:~$ source sandbox/bin/activate #enter sandbox
foo@bar:~$ python ./A02-program.py #launch project
foo@bar:~$ deactivate #exit sandbox
```

## Data Sources:

[Metropolitan Police Service, Recorded Crime: Borough Rates](https://data.london.gov.uk/dataset/recorded_crime_rates)

[Ministry of Housing, Communities & Local Government (MHCLG), Homelessness Provision, Borough](https://data.london.gov.uk/dataset/homelessness)

[Land Registry, Average House Prices by Borough, Ward, MSOA & LSOA](https://data.london.gov.uk/dataset/average-house-prices)

[Department for Education, GCSE Results by Borough](https://data.london.gov.uk/dataset/gcse-results-by-borough)

[Office for National Statistics (ONS), Qualifications of Working Age Population (NVQ), Borough](https://data.london.gov.uk/dataset/qualifications-working-age-population-nvq-borough)

[Greater London Authority (GLA), Household Income Estimates for Small Areas](https://data.london.gov.uk/dataset/household-income-estimates-small-areas)

[Office for National Statistics (ONS), Business Demographics and Survival Rates, Borough](https://data.london.gov.uk/dataset/business-demographics-and-survival-rates-borough)

[Office for National Statistics (ONS), Workless Households, Borough](https://data.london.gov.uk/dataset/workless-households-borough)

[Department of Health, Prevalence of Childhood Obesity, Borough, Ward and MSOA](https://data.london.gov.uk/dataset/prevalence-childhood-obesity-borough)

[Office for National Statistics (ONS), Life Expectancy at Birth and Age 65 by Ward](https://data.london.gov.uk/dataset/life-expectancy-birth-and-age-65-ward)

[HM Revenue & Customs, Children in Poverty, Borough and Ward](https://data.london.gov.uk/dataset/children-poverty-borough)

[Department of Health, Immunisation Rates for Children at 1st, 2nd and 5th Birthdays](https://data.london.gov.uk/dataset/immunisation-rates-children-1st-2nd-and-5th-birthdays )

[Office for National Statistics (ONS), Ratio of House Prices to Earnings, Borough](https://data.london.gov.uk/dataset/ratio-house-prices-earnings-borough)

[Office Of National Statistics, National Statistics Postcode Lookup UK](https://opendata.camden.gov.uk/Maps/National-Statistics-Postcode-Lookup-UK/tr8t-gqz7)
