import datetime
import json
import os

class ScheduleManager:
    def __init__(self, db_file="schedule_db.json"):
        self.db_file = db_file
        self.schedules = self.load_schedules()
    
    def load_schedules(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}
    
    def save_schedules(self):
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        with open(self.db_file, 'w', encoding='utf-8') as f:
            json.dump(self.schedules, f, ensure_ascii=False, indent=2)
    
    def add_schedule(self, time, location, event, duration):
        try:
            datetime.datetime.strptime(time, "%Y-%m-%d %H:%M")
        except ValueError:
            raise ValueError("Invalid time format. Please use YYYY-MM-DD HH:MM format")
        
        schedule_id = f"sch_{len(self.schedules) + 1}"
        self.schedules[schedule_id] = {
            "time": time,
            "location": location,
            "event": event,
            "duration": duration,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_schedules()
        return f"I've added a schedule for you with the following details:\nID: {schedule_id}\ntime: {time}\nlocation: {location}\nevent: {event}\nduration: {duration}"
    
    def delete_schedule(self, schedule_id):
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self.save_schedules()
            return f"Schedule {schedule_id} has been deleted"
        raise ValueError(f"No schedule found with ID: {schedule_id}")
    
    def update_schedule(self, schedule_id, **kwargs):
        if schedule_id not in self.schedules:
            raise ValueError(f"No schedule found with ID: {schedule_id}")
        
        allowed_fields = ["time", "location", "event", "duration"]
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == "time":
                    try:
                        datetime.datetime.strptime(value, "%Y-%m-%d %H:%M")
                    except ValueError:
                        raise ValueError("Invalid time format. Please use YYYY-MM-DD HH:MM format")
                self.schedules[schedule_id][key] = value
        
        self.save_schedules()
        return f"Schedule {schedule_id} has been updated"
    
    def query_schedules(self, time=None, location=None, event=None):
        results = []
        for schedule_id, details in self.schedules.items():
            match = True
            if time and details["time"] != time:
                match = False
            if location and details["location"] != location:
                match = False
            if event and event not in details["event"]:
                match = False
            if match:
                results.append({** details, "id": schedule_id})
        
        if not results:
            return "No relevant information was found"
        return results
    
    def get_all_schedules(self):
        all_schedules = [{"id": sid, **details} for sid, details in self.schedules.items()]
        return sorted(all_schedules, key=lambda x: x["time"])


def schedule_api(operation, **params):
    manager = ScheduleManager()
    
    if operation == "add":
        required_params = ["time", "location", "event", "duration"]
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        return manager.add_schedule(
            time=params["time"],
            location=params["location"],
            event=params["event"],
            duration=params["duration"]
        )
    
    elif operation == "delete":
        if "schedule_id" not in params:
            raise ValueError("Missing required parameter: schedule_id")
        return manager.delete_schedule(schedule_id=params["schedule_id"])
    
    elif operation == "update":
        if "schedule_id" not in params:
            raise ValueError("Missing required parameter: schedule_id")
        update_params = {k: v for k, v in params.items() if k != "schedule_id"}
        return manager.update_schedule(
            schedule_id=params["schedule_id"],
            **update_params
        )
    
    elif operation == "query":
        return manager.query_schedules(
            time=params.get("time"),
            location=params.get("location"),
            event=params.get("event")
        )
    
    elif operation == "get_all":
        return manager.get_all_schedules()
    
    else:
        raise ValueError(f"Unsupported operation: {operation}")

if __name__ == "__main__":
    try:
        schedule_api("add", time="2024-06-10 09:30")
    except Exception as e:
        print(f"Captured Error: {e}")

    add_result = schedule_api(
        "add",
        time="2024-06-10 09:30",
        location="Office 501",
        event="Product Review",
        duration="1.5 hours"
    )
    print("Add result:", add_result)