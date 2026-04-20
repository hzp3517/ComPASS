import os
import json
import random
import time
from typing import List, Dict
from openai import OpenAI

def onechat_gpt4o(system_prompt, user_prompt, model='gpt-4o'):
    """Function to call the onechat API"""
    api_key = "sk-xxx"  # onechat_key
    base_url = ''  # your base url for openai api, e.g. "https://api.openai.com/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    success = True
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=8192
        )
        output = response.choices[0].message.content
    except Exception as ex:
        success = False
        output = str(ex)
    return success, output


def shopping_assistant_api(item_type: str, item_name: str, item_features: List[str]) -> List[str]:
    """
    Directly callable shopping assistant API function
    
    Parameters:
        item_type: The category of the item (e.g., Electronics, Home Goods)
        item_name: The name of the item (e.g., Smartphone, Insulated Mug)
        item_features: List of item features (e.g., ["Wireless charging", "Waterproof"])
        
    Returns:
        List containing two strings:
        - Product details (name, type, description, features, price, rating, etc.)
        - Order confirmation information (order ID, store, date, amount, logistics, etc.)
    """
    # Initialize required data structures
    stores = ["Taobao Supermarket", "JD Mall", "Pinduoduo", "Suning.com", "Vipshop", "Dangdang.com", "Gome Online"]
    couriers = ["SF Express", "ZTO Express", "YTO Express", "STO Express", "Yunda Express", "JD Logistics"]
    payment_methods = ["Alipay", "WeChat Pay", "Bank Card Payment", "Cash on Delivery"]

    # 1. Generate product information
    def _generate_product_prompt():
        features_str = ", ".join(item_features)
        return f"""Please generate product information for a {item_name} in the {item_type} category with the following features: {features_str}.
Return strictly in the following JSON format without any additional content:
{{
    "name": "Product name (including brand)",
    "description": "Detailed description of product features and functions",
    "price": price (number, in yuan),
    "rating": rating (number between 1-5, one decimal place),
    "review_count": number of reviews (integer),
    "features": ["feature1", "feature2", "feature3", ...]
}}"""

    def _get_fallback_product():
        return {
            "name": f"Default Brand {item_name}",
            "description": f"This is a high-quality {item_name} in the {item_type} category, with features including {', '.join(item_features)}, reliable quality.",
            "price": round(random.uniform(50, 2000), 2),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "review_count": random.randint(10, 10000),
            "features": item_features + ["Reliable quality", "After-sales guarantee", "High cost performance"],
            "type": item_type
        }

    # Get product data
    system_prompt = """You are a product information generation expert, capable of generating reasonable product information based on user needs.
The generated information must comply with market rules, have reasonable prices, detailed descriptions, and clear features. Please return strictly in the specified JSON format."""
    user_prompt = _generate_product_prompt()
    success, response = onechat_gpt4o(system_prompt, user_prompt)
    
    try:
        if success:
            product = json.loads(response)
            product["type"] = item_type
        else:
            product = _get_fallback_product()
    except json.JSONDecodeError:
        product = _get_fallback_product()
    except Exception:
        product = _get_fallback_product()

    # 2. Generate product details string
    features_str = ", ".join(product["features"])
    product_details = (f"Product Name: {product['name']}\n"
                      f"Type: {product['type']}\n"
                      f"Description: {product['description']}\n"
                      f"Features: {features_str}\n"
                      f"Price: ¥{product['price']:.2f}\n"
                      f"Rating: {product['rating']}/5.0 ({product['review_count']} reviews)")

    # 3. Generate order confirmation
    def _generate_order_prompt():
        store = random.choice(stores)
        return f"""Please generate online shopping order confirmation information for the following product:
Product name: {product['name']}
Price: {product['price']} yuan
Purchasing platform: {store}

Return strictly in the following JSON format without any additional content:
{{
    "order_id": "Order number",
    "purchase_date": "Purchase date (format: YYYY-MM-DD)",
    "quantity": Purchase quantity (integer),
    "total_amount": Total amount (number),
    "payment_method": "Payment method",
    "shipping_info": {{
        "courier": "Express company",
        "estimated_delivery": "Estimated delivery time"
    }},
    "status": "Order status"
}}"""

    def _get_fallback_order():
        store = random.choice(stores)
        return {
            "order_id": f"ORD{random.randint(10000000, 99999999)}",
            "product": {
                "name": product["name"],
                "price": product["price"]
            },
            "store": store,
            "purchase_date": time.strftime('%Y-%m-%d', time.localtime()),
            "quantity": 1,
            "total_amount": product["price"],
            "payment_method": random.choice(payment_methods),
            "shipping_info": {
                "courier": random.choice(couriers),
                "estimated_delivery": f"Within {random.randint(2, 7)} days"
            },
            "status": "Confirmed"
        }

    # Get order data
    order_system_prompt = """You are an order information generation expert, capable of generating reasonable order confirmation information based on product information.
The generated information must comply with online shopping procedures and include necessary order elements. Please return strictly in the specified JSON format."""
    order_user_prompt = _generate_order_prompt()
    order_success, order_response = onechat_gpt4o(order_system_prompt, order_user_prompt)
    
    try:
        if order_success:
            order = json.loads(order_response)
            order["product"] = {
                "name": product["name"],
                "price": product["price"]
            }
            order["store"] = [s for s in stores if s in order_user_prompt][0]
        else:
            order = _get_fallback_order()
    except json.JSONDecodeError:
        order = _get_fallback_order()
    except Exception:
        order = _get_fallback_order()

    # 4. Generate order details string
    order_details = (f"Order Confirmation\n"
                    f"Order ID: {order['order_id']}\n"
                    f"Product: {order['product']['name']}\n"
                    f"Purchased from: {order['store']}\n"
                    f"Purchase Date: {order['purchase_date']}\n"
                    f"Quantity: {order['quantity']}\n"
                    f"Total Amount: ¥{order['total_amount']:.2f}\n"
                    f"Payment Method: {order['payment_method']}\n"
                    f"Courier: {order['shipping_info']['courier']}\n"
                    f"Estimated Delivery: {order['shipping_info']['estimated_delivery']}\n"
                    f"Order Status: {order['status']}")

    return product_details+order_details


# Example usage
if __name__ == "__main__":
    # Example 1: Get information for a smartphone
    phone_features = ["6.7-inch screen", "5000mAh battery", "100MP camera", "5G network"]
    result = shopping_assistant_api("Electronics", "Smartphone", phone_features)
    print(result)


    # Example 2: Get information for an insulated mug
    mug_features = ["316 stainless steel", "500ml capacity", "Vacuum insulation"]
    result = shopping_assistant_api("Home Goods", "Insulated Mug", mug_features)
    print(result)
