.. py:module:: colliderscope

CLI Reference
=============

The following CLI provides a web interface to interactive with HEP data.
In order to use it, you must install ``colliderscope`` with the optional
``webui`` dependencies, *ie.*

.. code-block:: bash

   pip install colliderscope[webui]


After running this, it should be possible to execute the following commands
from your terminal without exceptions.

.. click:: colliderscope.__main__:main
  :prog: colliderscope
  :nested: full
