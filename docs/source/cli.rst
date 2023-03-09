.. py:module:: colliderscope

CLI Reference
=============

The following CLI provides a web interface to interactive with HEP data.
In order to use it, you must install ``colliderscope`` with the optional
``webui`` dependencies, *ie.*

.. code-block:: bash

   pip install "colliderscope[webui]"


After running this, it should be possible to execute the following commands
from your terminal without exceptions.

.. note::
   This assumes you already have ``pythia8``, compiled with the Python
   interface, installed in your development environment. If you do not, you
   can `install it via conda <https://anaconda.org/conda-forge/pythia8>`_
   from the ``conda-forge`` channel. See the
   `showerpipe <https://showerpipe.readthedocs.io/>`_ documentation for more
   information.

.. click:: colliderscope.__main__:main
  :prog: colliderscope
  :nested: full
