Usage
=====

pybear is a set of data analytics tools that use the `scikit-learn <https://scikit-learn.org/stable/index.html>`_
API. It is designed to enhance your data analytics experience when used in conjunction 
with other packages that follow that same API. Some packages that follow that API include:

`xgboost <https://xgboost.readthedocs.io/en/stable/>`_

`lightgbm <https://lightgbm.readthedocs.io/en/latest/index.html>`_

`imbalanced-learn <https://imbalanced-learn.org/stable/install.html/>`_

pybear drops right into workflows that use those packages. For example, all pybear 
classes can be incorporated into a scikit `pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>`_.

Check out the :ref:`installation <installation>` section for information on how to 
install pybear.

The folder structure of pybear is nearly identical to scikit-learn. This is so
those that are familiar with the scikit layout and have experience with writing
the associated import statements have an easy transition to pybear. The pybear
subfolders are *base*, *feature_extraction*, *model_selection*, *new_numpy*,
*preprocessing*, and *utilities*.

You can import pybear's packages in the same way you would with scikit. Here
are a few examples of how you could import and use pybear modules:

.. code-block:: console

    from pybear.preprocessing import InterceptManager as IM

    trfm = IM()
    trfm.fit(X, y)

    from pybear import preprocessing as pp

    trfm = pp.ColumnDeduplicateTransformer()
    trfm.fit(X, y)

See the :ref:`API <api>` section of the docs for a complete listing of the
available classes.
