import random

def get_education_by_age(age):
    if age < 6:
        return "No formal education"
    elif 6 <= age <= 11:
        options = [
            "Primary school (incomplete)",
            "No formal education"
        ]
    elif 12 <= age <= 15:
        options = [
            "Lower secondary school (attending)"
        ]
    elif 16 <= age <= 18:
        options = [
            "Upper secondary school (attending)",
            "Vocational high school (attending)",
            "Technical school (attending)"
        ]
    elif 19 <= age <= 22:
        options = [
            "Upper secondary school graduate",
            "Vocational high school graduate",
            "Technical school graduate",
            "Associate degree (in progress)",
            "Associate degree holder",
            "Bachelor's degree (in progress)"
        ]
    elif 23 <= age <= 25:
        options = [
            "Upper secondary school graduate",
            "Vocational high school graduate",
            "Technical school graduate",
            "Associate degree (in progress)",
            "Associate degree holder",
            "Bachelor's degree (in progress)",
            "Bachelor's degree holder",
            "Master's degree (in progress)",
            "Master's degree holder",
            "Doctoral degree (in progress)"
        ]
    elif 26 <= age <= 30:
        options = [
            "Upper secondary school graduate",
            "Vocational high school graduate",
            "Technical school graduate",
            "Associate degree (in progress)",
            "Associate degree holder",
            "Bachelor's degree (in progress)",
            "Bachelor's degree holder",
            "Master's degree (in progress)",
            "Master's degree holder",
            "Doctoral degree (in progress)",
            "Doctoral degree holder"
        ]
    else:  
        options = [
            "Upper secondary school graduate",
            "Vocational high school graduate",
            "Technical school graduate",
            "Bachelor's degree holder",
            "Master's degree holder",
            "Doctoral degree holder",
            "Adult education (attending)",
            "Adult education graduate",
            "Distance learning certificate",
            "Professional training certificate",
            "Continuing education courses"
        ]
    
    return random.choice(options)


if __name__ == "__main__":
    test_ages = [3, 8, 14, 17, 20, 24, 28, 35, 50]
    for age in test_ages:
        print(f" {get_education_by_age(age)}")