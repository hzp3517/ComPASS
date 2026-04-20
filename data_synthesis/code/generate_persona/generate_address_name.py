import requests
from faker import Faker


def generate_american_info(gender=None):
    fake = Faker("en_US")  

    try:
        if gender == "male":
            first_name = fake.first_name_male()
        elif gender == "female":
            first_name = fake.first_name_female()
        else:
            first_name = fake.first_name() 
        last_name = fake.last_name()


        address = fake.address().replace("\n", ", ")


        area_code = fake.numerify("###")  
        central_office_code = fake.numerify("###")  
        line_number = fake.numerify("####")  
        phone_international = f"+1-{area_code}-{central_office_code}-{line_number}"

        return True, first_name, last_name, address, phone_international

    except requests.exceptions.RequestException as e:
        print(f"failed: {e}")
        return False, None, None, None, None
    except Exception as e:
        print(f"error: {e}")
        return False, None, None, None, None


# 使用示例
if __name__ == "__main__":
    pass