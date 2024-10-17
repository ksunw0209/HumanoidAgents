import json
import random
import yaml

from collections import defaultdict
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.parser import ParserError

def load_json_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def write_json_file(data, filename):
    with open(filename, "w", encoding="utf-8") as fw:
        fw.write(json.dumps(data, indent=4))

def bucket_agents_by_location(list_of_agent_statuses, agents):
    location_to_agents = defaultdict(list)
    for agent_status, agent in zip(list_of_agent_statuses, agents):
        agent_location = agent_status['location']
        # to make agent location hashable
        tuple_agent_location = tuple(agent_location)
        location_to_agents[tuple_agent_location].append(agent)
    return location_to_agents

def get_pairwise_conversation_by_agents_in_same_location(location_to_agents, curr_time):
    # only 1 conversation per location, when there are 2 or more agents

    location_to_conversations = defaultdict(list)
    for location, agents in location_to_agents.items():
        if len(agents) > 1:
            selected_agents = random.sample(agents, 2)
            initiator, responder = selected_agents
            convo_history = initiator.dialogue(responder, curr_time)
            location_to_conversations[location].append(convo_history)
    return location_to_conversations

def override_agent_kwargs_with_condition(kwargs, condition):
    if condition is None:
        return kwargs
    #emotion
    elif condition in ["disgusted", "afraid", "sad", "surprised", "happy", "angry", "neutral"]:
        kwargs["emotion"] = condition
    #set each social relationship closeness to specific value
    elif "closeness" in condition:
        for key in kwargs["social_relationships"]:
            kwargs["social_relationships"][key]['closeness'] = int(condition.split("_")[1])
    #basic needs
    elif condition in ["fullness", "social", "fun", "health", "energy"]:
        if "basic_needs" not in kwargs:
            kwargs["basic_needs"] = {}
        for condition_i, basic_need in enumerate(kwargs["basic_needs"]):
            if basic_need['name'] == condition:
                kwargs["basic_needs"][condition_i]["start_value"] = 0
    else:
        raise ValueError("condition is not valid")
    return kwargs

def get_curr_time_to_daily_event(daily_events_filename):
    curr_time_to_daily_event = defaultdict(None)

    if daily_events_filename is not None:
        with open(daily_events_filename, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
        for date, event in loaded_yaml.items():
            time = "12:00 am"
            curr_time = DatetimeNL.convert_nl_datetime_to_datetime(date, time)
            curr_time_to_daily_event[curr_time] = event
    return curr_time_to_daily_event


class DatetimeNL:

    @staticmethod
    def get_date_nl(curr_time):
        # e.g. Monday Jan 02 2023
        day_of_week = curr_time.strftime('%A')
        month_date_year = curr_time.strftime("%b %d %Y")
        date = f"{day_of_week} {month_date_year}"
        return date

    @staticmethod
    def get_time_nl(curr_time):
        #e.g. 12:00 am and 07:00 pm (note there is a leading zero for 7pm)
        time = curr_time.strftime('%I:%M %p').lower()
        return time

    @staticmethod
    def convert_nl_datetime_to_datetime(date, time):
        # missing 0 in front of time
        if len(time) != len("12:00 am"):
            time = "0" + time.upper()
        
        date_string = date + ' ' + time
        original_error = None
        try:
            # First, try parsing with the original format
            return datetime.strptime(date_string, "%A %b %d %Y %I:%M %p")
        except ValueError as e:
            original_error = str(e)
            try:
                # If that fails, try parsing with a 24-hour format
                return datetime.strptime(date_string, "%A %b %d %Y %H:%M")
            except ValueError:
                try:
                    # If both fail, use a more flexible parsing method
                    return parser.parse(date_string)
                except ParserError:
                    # If dateutil parser fails, try a custom correction
                    corrected_string = date_string.replace(" pm", "").replace(" am", "")
                    try:
                        return datetime.strptime(corrected_string, "%A %b %d %Y %H:%M")
                    except ValueError:
                        # If all parsing attempts fail, raise a custom error with the original error message
                        raise ValueError(f"Unable to parse date string: {date_string}. Original error: {original_error}")
        return curr_time

    def subtract_15_min(curr_time):
        return curr_time - timedelta(minutes=15)
    
    def add_15_min(curr_time):
        return curr_time + timedelta(minutes=15)
        
    @staticmethod
    def get_formatted_date_time(curr_time):
        # e.g. "It is Monday Jan 02 2023 12:00 am"
        date_in_nl = DatetimeNL.get_date_nl(curr_time)
        time_in_nl = DatetimeNL.get_time_nl(curr_time)
        formatted_date_time = f"It is {date_in_nl} {time_in_nl}"
        return formatted_date_time
    
    @staticmethod
    def get_date_range(start_date, end_date):
        """
        Get date range between start_date (inclusive) and end_date (inclusive)
        
        start_date and end_date are str in the format YYYY-MM-DD
        """
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_range = []
        
        while start_date <= end_date:
            date_range.append(start_date.strftime('%Y-%m-%d'))
            start_date += timedelta(days=1)
        if not date_range:
            raise ValueError("end_date must be later or equal to start_date")
        return date_range
