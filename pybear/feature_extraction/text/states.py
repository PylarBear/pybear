# Author:
#         Bill Sousa
#
# License: BSD 3 clause
#



def states_incl_dc():
    return ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
            'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA',
            'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
             'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
             'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']

def states():
    return [state for state in states_incl_dc() if state != 'DC']

def lower_states():
    return [state for state in states_incl_dc() if state not in ['DC','HI','AK']]



