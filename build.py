#   -*- coding: utf-8 -*-
from pybuilder.core import use_plugin, init

use_plugin("python.core")
use_plugin("python.install_dependencies")
use_plugin("python.unittest")
use_plugin("python.flake8")
use_plugin("python.coverage")
use_plugin("python.distutils")


name = "pydewey"
default_task = ['install_dependencies', 'analyze', 'publish']


@init
def set_properties(project):
    # project.set_property("flake8_break_build", True)
    project.set_property("flake8_verbose_output", True)
    project.set_property("flake8_include_test_sources", True)

    project.set_property("ut_coverage_branch_threshold_warn", 80)
    project.set_property("ut_coverage_branch_partial_threshold_warn", 80)
    project.set_property('coverage_break_build', False)
