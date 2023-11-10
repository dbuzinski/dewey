#   -*- coding: utf-8 -*-
from pybuilder.core import Author, use_plugin, init

use_plugin("python.core")
use_plugin("python.install_dependencies")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "pydewey"
summary = "Dewey — a fast reproducible training automation tool for MLOps pipelines."
description = """Dewey — a fast reproducible training automation tool for MLOps pipelines.

Dewey is a machine learning automation tool written to create consistent reproducible ways to train models 
in a framework agnostic way. It allows providing a training specification, and the Dewey training framework 
takes care of all of the standard boilerplate code involving writing training loops, monitoring & metrics, 
managing model checkpoints, and more. Please note that this tool is in early stages of development and is 
prone to rapid updates and breaking API changes.
"""

authors = [Author("David Buzinski", "davidbuzinski@gmail.com")]

maintainers = [Author("David Buzinski", "davidbuzinski@gmail.com")]

url = "https://github.com/dbuzinski/dewey"
urls = {"Bug Tracker": "https://github.com/dbuzinski/dewey/issues",
        "Source Code": "https://github.com/dbuzinski/dewey",
        }
license = "Apache License, Version 2.0"
version = "0.3.0.dev"

requires_python = ">=3.10"

default_task = ["analyze", "publish"]

@init
def set_properties(project):
    project.depends_on_requirements("requirements.txt")

    project.set_property("flake8_break_build", True)
    project.set_property("flake8_verbose_output", True)
    project.set_property("flake8_include_test_sources", True)

    project.set_property("ut_coverage_branch_threshold_warn", 80)
    project.set_property("ut_coverage_branch_partial_threshold_warn", 80)
    project.set_property('coverage_break_build', False)
