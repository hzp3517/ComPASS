import os
import json
import sqlite3
import datetime
import uuid
from openai import OpenAI


def dict_to_string(d, key_val_sep=": ", item_sep=", "):
    return item_sep.join([f"{k}{key_val_sep}{v}" for k, v in d.items()])


def onechat_gpt4o(system_prompt, user_prompt, model='gpt-4o'):
    api_key = "sk-xx"
    base_url = ''# your base url for openai api, e.g. "https://api.openai.com/v1"
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
            temperature=0,
            max_tokens=8192
        )
        output = response.choices[0].message.content
    except Exception as ex:
        success = False
        output = str(ex)

    return success, output


class AppointmentDatabase:
    def __init__(self, db_name="/xxx/.../medical_appointments.db"):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS appointments (
            id TEXT PRIMARY KEY,
            user_name TEXT NOT NULL,
            user_phone TEXT NOT NULL,
            user_id TEXT,
            illness_description TEXT NOT NULL,
            desired_date TEXT NOT NULL,
            hospital TEXT,
            department TEXT,
            doctor TEXT,
            appointment_date TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ''')
        self.conn.commit()

    def add_appointment(self, appointment):
        cursor = self.conn.cursor()
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        appointment['created_at'] = current_time
        appointment['updated_at'] = current_time

        cursor.execute('''
        INSERT INTO appointments (
            id, user_name, user_phone, user_id, illness_description, desired_date,
            hospital, department, doctor, appointment_date, status, created_at, updated_at
        ) VALUES 
        ''', (
            appointment['id'],
            appointment['user_name'],
            appointment['user_phone'],
            appointment.get('user_id', ''),
            appointment['illness_description'],
            appointment['desired_date'],
            appointment['hospital'],
            appointment['department'],
            appointment['doctor'],
            appointment['appointment_date'],
            appointment.get('status', 'confirmed'),
            appointment['created_at'],
            appointment['updated_at']
        ))
        self.conn.commit()
        return True

    def get_appointment(self, appointment_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM appointments WHERE id = ?", (appointment_id,))
        result = cursor.fetchone()
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None

    def update_appointment(self, appointment_id, updates):
        if not updates:
            return False

        updates['updated_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [appointment_id]

        cursor = self.conn.cursor()
        cursor.execute(f"UPDATE appointments SET {set_clause} WHERE id = ?", values)
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_appointment(self, appointment_id):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM appointments WHERE id = ?", (appointment_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def search_appointments(self, **kwargs):
        if not kwargs:
            return []

        where_clause = " AND ".join([f"{key} LIKE ?" for key in kwargs.keys()])
        values = [f"%{value}%" for value in kwargs.values()]

        cursor = self.conn.cursor()
        cursor.execute(f"SELECT * FROM appointments WHERE {where_clause}", values)
        results = cursor.fetchall()

        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]

    def close(self):
        self.conn.close()


class MedicalAppointmentSystem:
    def __init__(self):
        self.db = AppointmentDatabase()
        self.system_prompt = """
        You are a professional medical appointment consultant. Based on the user's condition and desired time, recommend hospital, department, and doctor, and arrange an appointment.
        Requirements:
        1. Choose appropriate department
        2. Use real hospital names
        3. Respect desired time or provide closest available
        4. Doctor names should be realistic
        5. Return JSON only with:
           hospital, department, doctor, appointment_date
        """

    def _clean_llm_json(self, llm_response):
        if llm_response.startswith("```"):
            lines = llm_response.strip().split("\n")
            if len(lines) > 1:
                json_content = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
            else:
                json_content = ""
        else:
            json_content = llm_response.strip()

        return json_content.replace("\n", "").replace("  ", " ")

    def generate_appointment_info(self, illness_description, desired_date):
        user_prompt = f"Illness description: {illness_description}\nDesired appointment date: {desired_date}"
        success, response = onechat_gpt4o(self.system_prompt, user_prompt)

        if not success:
            return None, f"Generation failed: {response}"

        try:
            clean_json = self._clean_llm_json(response)
            appointment_info = json.loads(clean_json)

            required_fields = ['hospital', 'department', 'doctor', 'appointment_date']
            if all(field in appointment_info for field in required_fields):
                return appointment_info, None
            else:
                missing = [f for f in required_fields if f not in appointment_info]
                return None, f"Missing fields: {', '.join(missing)}"

        except json.JSONDecodeError as e:
            return None, f"JSON parse error: {str(e)}"

    def book_appointment(self, user_name, user_phone, illness_description, desired_date, user_id=""):
        appointment_id = str(uuid.uuid4())
        llm_info, error = self.generate_appointment_info(illness_description, desired_date)

        if error:
            return False, error

        appointment = {
            'id': appointment_id,
            'user_name': user_name,
            'user_phone': user_phone,
            'user_id': user_id,
            'illness_description': illness_description,
            'desired_date': desired_date,
            'hospital': llm_info['hospital'],
            'department': llm_info['department'],
            'doctor': llm_info['doctor'],
            'appointment_date': llm_info['appointment_date'],
            'status': 'confirmed'
        }

        if self.db.add_appointment(appointment):
            return True, {
                "message": "Appointment successful",
                "appointment_id": appointment_id,
                "hospital": llm_info['hospital'],
                "department": llm_info['department'],
                "doctor": llm_info['doctor'],
                "appointment_date": llm_info['appointment_date']
            }

        return False, "Save failed"

    def cancel_appointment(self, appointment_id):
        if self.db.get_appointment(appointment_id):
            if self.db.update_appointment(appointment_id, {'status': 'cancelled'}):
                return True, "Cancelled"
            return False, "Cancel failed"
        return False, "Not found"

    def modify_appointment(self, appointment_id, **updates):
        appointment = self.db.get_appointment(appointment_id)
        if not appointment:
            return False, "Not found"

        valid_fields = ['user_name', 'user_phone', 'user_id', 'desired_date', 'status']
        filtered_updates = {k: v for k, v in updates.items() if k in valid_fields}

        if not filtered_updates:
            return False, "No valid updates"

        if 'desired_date' in filtered_updates:
            llm_info, error = self.generate_appointment_info(
                appointment['illness_description'],
                filtered_updates['desired_date']
            )
            if error:
                return False, error

            filtered_updates.update({
                'hospital': llm_info['hospital'],
                'department': llm_info['department'],
                'doctor': llm_info['doctor'],
                'appointment_date': llm_info['appointment_date']
            })

        if self.db.update_appointment(appointment_id, filtered_updates):
            return True, "Updated"

        return False, "Update failed"

    def query_appointments(self, **kwargs):
        valid_fields = ['id', 'user_name', 'user_phone', 'user_id', 'status', 'hospital', 'department']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}

        results = self.db.search_appointments(**filtered_kwargs)
        if results == []:
            results = "No results"

        return True, results

    def close(self):
        self.db.close()


def medical_appointment_operation(operation, **kwargs):
    system = MedicalAppointmentSystem()

    try:
        if operation == "book":
            required = ["user_name", "user_phone", "illness_description", "desired_date"]
            if not all(k in kwargs for k in required):
                return {"success": False, "result": "Missing parameters"}

            success, result = system.book_appointment(
                kwargs["user_name"],
                kwargs["user_phone"],
                kwargs["illness_description"],
                kwargs["desired_date"],
                kwargs.get("user_id", "")
            )
            return result

        elif operation == "query":
            success, results = system.query_appointments(**kwargs)
            return {"success": success, "result": results}

        elif operation == "modify":
            if "appointment_id" not in kwargs:
                return {"success": False, "result": "Missing appointment_id"}

            updates = {k: v for k, v in kwargs.items() if k != "appointment_id"}
            success, result = system.modify_appointment(kwargs["appointment_id"], **updates)
            return {"success": success, "result": result}

        elif operation == "cancel":
            if "appointment_id" not in kwargs:
                return {"success": False, "result": "Missing appointment_id"}

            success, result = system.cancel_appointment(kwargs["appointment_id"])
            return {"success": success, "result": result}

        else:
            return {"success": False, "result": "Invalid operation"}

    finally:
        system.close()


def book_appointment(user_name, user_phone, illness_description, desired_date, user_id=""):
    return medical_appointment_operation(
        "book",
        user_name=user_name,
        user_phone=user_phone,
        illness_description=illness_description,
        desired_date=desired_date,
        user_id=user_id
    )


def query_appointments(**kwargs):
    return medical_appointment_operation("query", **kwargs)


def modify_appointment(appointment_id, **updates):
    params = {"appointment_id": appointment_id}
    params.update(updates)
    return medical_appointment_operation("modify", **params)


def cancel_appointment(appointment_id):
    return medical_appointment_operation("cancel", appointment_id=appointment_id)