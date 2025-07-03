{{ name }}
{{ "=" * name|length }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance: False
   :members:

{% if methods %}
.. rubric:: Methods

.. autosummary::
   :nosignatures:
{% for item in methods %}
   ~{{ name }}.{{ item.name }}
{% endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Attributes

.. autosummary::
   :nosignatures:
{% for item in attributes %}
   ~{{ name }}.{{ item.name }}
{% endfor %}
{% endif %}
