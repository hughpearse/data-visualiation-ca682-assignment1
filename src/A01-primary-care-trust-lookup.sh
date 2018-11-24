#!/bin/bash

function main() {
    for primary_care_trust_code in $(tail -n +2 ./immunization-1st-birthday.csv  | cut -d ',' -f1);
    do
        if [[ "${primary_care_trust_code}" != *"E16"* ]] && [[ "${primary_care_trust_code}" != *"E17"* ]];
        then
            echo ",${primary_care_trust_code}"
        else
            local borough_code=$(grep -m1 "${primary_care_trust_code}" ./../National_Statistics_Postcode_Lookup_UK.csv | cut -d ',' -f11)
            echo "${borough_code},${primary_care_trust_code}"
        fi
    done
}
main
