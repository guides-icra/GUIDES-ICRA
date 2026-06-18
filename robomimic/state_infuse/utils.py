import re


def parse_next_action(text):
    # Regular expression to match "Next Action" whether or not it's surrounded by **
    match = re.search(r'Next Action\s*:\s*(.*?)(?:\s*(?=\n|$|\*\*|Potential Error))', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


if __name__ == '__main__':
    parse_next_action(""" pick the ketchup from the cabinet and place it on the counter : State: Grasping the ketchup bottle  
Next Action: Lift the ketchup bottle from the cabinet  
Potential Error: Misalignment causing the bottle to drop """)