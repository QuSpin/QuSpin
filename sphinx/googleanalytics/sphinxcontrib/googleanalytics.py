#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sphinx.errors import ExtensionError


def add_ga_javascript(app, pagename, templatename, context, doctree):
    if not app.config.googleanalytics_enabled:
        return

    metatags = context.get('metatags', '')
    metatags += """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=%s"></script>""" % app.config.googleanalytics_id
    metatags += """
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', '%s');
    </script>
    """ % app.config.googleanalytics_id
    context['metatags'] = metatags


def check_config(app):
    if not app.config.googleanalytics_id:
        raise ExtensionError("'googleanalytics_id' config value must be set for ga statistics to function properly.")


def setup(app):
    app.add_config_value('googleanalytics_id', '', 'html')
    app.add_config_value('googleanalytics_enabled', True, 'html')
    app.connect('html-page-context', add_ga_javascript)
    app.connect('builder-inited', check_config)
    return {
      'version': '0.3',
      'parallel_read_safe': True,
      'parallel_write_safe': True,
      }
