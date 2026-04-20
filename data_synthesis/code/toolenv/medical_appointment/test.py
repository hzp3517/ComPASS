import time
import uuid

# 导入医疗预约系统的核心函数
from  api import (  # 请将 your_module_name 替换为实际的文件名（不含.py）
    book_appointment,
    query_appointments,
    modify_appointment,
    cancel_appointment,
    medical_appointment_operation
)

def test_all_functions():
    """测试所有功能，直接输出调用结果"""
    print("=" * 60)
    print("开始测试医疗预约系统所有功能")
    print("=" * 60)

    # 测试1：预约功能（book）
    print("\n【测试1：预约功能】")
    book_params = {
        "user_name": "张三",
        "user_phone": "13800138000",
        "illness_description": "感冒发烧，咳嗽持续3天",
        "desired_date": "2025-11-15",
        "user_id": "330106199001011234"
    }
    book_result = book_appointment(**book_params)
    print(f"调用参数：{book_params}")
    print(f"返回结果：{book_result}")
    
    '''# 提取预约ID用于后续测试
    appointment_id = book_result["result"]["appointment_id"] if book_result["success"] else None
    print(f"提取的预约ID：{appointment_id}")

    # 测试2：查询功能（query）- 按用户名查询
    print("\n" + "-" * 50)
    print("【测试2：查询功能 - 按用户名查询】")
    query_params = {"user_name": "张三"}
    query_result = query_appointments(**query_params)
    print(f"调用参数：{query_params}")
    print(f"返回结果：{query_result}")

    # 测试3：查询功能（query）- 按预约ID查询
    print("\n" + "-" * 50)
    print("【测试3：查询功能 - 按预约ID查询】")
    if appointment_id:
        query_id_params = {"id": appointment_id}
        query_id_result = query_appointments(**query_id_params)
        print(f"调用参数：{query_id_params}")
        print(f"返回结果：{query_id_result}")
    else:
        print("跳过：预约失败，无可用预约ID")

    # 测试4：修改功能（modify）- 修改预约时间和用户名
    print("\n" + "-" * 50)
    print("【测试4：修改功能 - 修改预约时间和用户名】")
    if appointment_id:
        modify_params = {
            "appointment_id": appointment_id,
            "desired_date": "2025-11-16",
            "user_name": "张三丰"
        }
        modify_result = modify_appointment(**modify_params)
        print(f"调用参数：{modify_params}")
        print(f"返回结果：{modify_result}")
    else:
        print("跳过：预约失败，无法修改")

    # 测试5：查询修改后的结果
    print("\n" + "-" * 50)
    print("【测试5：查询修改后的预约信息】")
    if appointment_id:
        query_modify_params = {"id": appointment_id}
        query_modify_result = query_appointments(**query_modify_params)
        print(f"调用参数：{query_modify_params}")
        print(f"返回结果：{query_modify_result}")
    else:
        print("跳过：预约失败，无法查询修改结果")

    # 测试6：取消功能（cancel）
    print("\n" + "-" * 50)
    print("【测试6：取消功能】")
    if appointment_id:
        cancel_result = cancel_appointment(appointment_id)
        print(f"调用参数：appointment_id={appointment_id}")
        print(f"返回结果：{cancel_result}")
    else:
        print("跳过：预约失败，无法取消")

    # 测试7：查询已取消的预约
    print("\n" + "-" * 50)
    print("【测试7：查询已取消的预约】")
    if appointment_id:
        query_cancel_params = {"id": appointment_id, "status": "cancelled"}
        query_cancel_result = query_appointments(**query_cancel_params)
        print(f"调用参数：{query_cancel_params}")
        print(f"返回结果：{query_cancel_result}")
    else:
        print("跳过：预约失败，无法查询取消状态")

    # 测试8：错误测试 - 取消不存在的预约
    print("\n" + "-" * 50)
    print("【测试8：错误测试 - 取消不存在的预约】")
    invalid_appointment_id = str(uuid.uuid4())
    cancel_invalid_result = cancel_appointment(invalid_appointment_id)
    print(f"调用参数：appointment_id={invalid_appointment_id}")
    print(f"返回结果：{cancel_invalid_result}")'''



if __name__ == "__main__":
    test_all_functions()