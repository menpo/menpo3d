from collections import OrderedDict
from time import sleep
from IPython import get_ipython
from ipywidgets import Box
import ipywidgets
from traitlets.traitlets import List

# The below classes have been copied from
# the deprecated menpowidgets package
# MenpoWidget can be found in abstract.py
# LinearModelParametersWidget in options.py
class MenpoWidget(Box):
    r"""
    Base class for defining a Menpo widget.

    The widget has a `selected_values` trait that can be used in order to
    inspect any changes that occur to its children. It also has functionality
    for adding, removing, replacing or calling the handler callback function of
    the `selected_values` trait.

    Parameters
    ----------
    children : `list` of `ipywidgets`
        The `list` of `ipywidgets` objects to be set as children in the
        `ipywidgets.Box`.
    trait : `traitlets.TraitType` subclass
        The type of the `selected_values` object that gets added as a trait
        in the widget. Possible options from `traitlets` are {``Int``, ``Float``,
        ``Dict``, ``List``, ``Tuple``}.
    trait_initial_value : `int` or `float` or `dict` or `list` or `tuple`
        The initial value of the `selected_values` trait.
    render_function : `callable` or ``None``, optional
        The render function that behaves as a callback handler of the
        `selected_values` trait for the `change` event. Its signature can be
        ``render_function()`` or ``render_function(change)``, where ``change``
        is a `dict` with the following keys:

        - ``owner`` : the `HasTraits` instance
        - ``old`` : the old value of the modified trait attribute
        - ``new`` : the new value of the modified trait attribute
        - ``name`` : the name of the modified trait attribute.
        - ``type`` : ``'change'``

        If ``None``, then nothing is added.
    """
    def __init__(self, children, trait, trait_initial_value,
                 render_function=None):
        # Create box object
        super(MenpoWidget, self).__init__(children=children)

        # Add trait for selected values
        selected_values = trait(default_value=trait_initial_value)
        selected_values_trait = {'selected_values': selected_values}
        self.add_traits(**selected_values_trait)
        self.selected_values = trait_initial_value

        # Set render function
        self._render_function = None
        self.add_render_function(render_function)

    def add_render_function(self, render_function):
        r"""
        Method that adds the provided `render_function()` as a callback handler
        to the `selected_values` trait of the widget. The given function is
        also stored in `self._render_function`.

        Parameters
        ----------
        render_function : `callable` or ``None``, optional
            The render function that behaves as a callback handler of the
            `selected_values` trait for the `change` event. Its signature can be
            ``render_function()`` or ``render_function(change)``, where
            ``change`` is a `dict` with the following keys:

            - ``owner`` : the `HasTraits` instance
            - ``old`` : the old value of the modified trait attribute
            - ``new`` : the new value of the modified trait attribute
            - ``name`` : the name of the modified trait attribute.
            - ``type`` : ``'change'``

            If ``None``, then nothing is added.
        """
        self._render_function = render_function
        if self._render_function is not None:
            self.observe(self._render_function, names='selected_values',
                         type='change')

    def remove_render_function(self):
        r"""
        Method that removes the current `self._render_function()` as a callback
        handler to the `selected_values` trait of the widget and sets
        ``self._render_function = None``.
        """
        if self._render_function is not None:
            self.unobserve(self._render_function, names='selected_values',
                           type='change')
            self._render_function = None

    def replace_render_function(self, render_function):
        r"""
        Method that replaces the current `self._render_function()` with the
        given `render_function()` as a callback handler to the `selected_values`
        trait of the widget.

        Parameters
        ----------
        render_function : `callable` or ``None``, optional
            The render function that behaves as a callback handler of the
            `selected_values` trait for the `change` event. Its signature can be
            ``render_function()`` or ``render_function(change)``, where
            ``change`` is a `dict` with the following keys:

            - ``owner`` : the `HasTraits` instance
            - ``old`` : the old value of the modified trait attribute
            - ``new`` : the new value of the modified trait attribute
            - ``name`` : the name of the modified trait attribute.
            - ``type`` : ``'change'``

            If ``None``, then nothing is added.
        """
        # remove old function
        self.remove_render_function()

        # add new function
        self.add_render_function(render_function)

    def call_render_function(self, old_value, new_value, type_value='change'):
        r"""
        Method that calls the existing `render_function()` callback handler.

        Parameters
        ----------
        old_value : `int` or `float` or `dict` or `list` or `tuple`
            The old `selected_values` value.
        new_value : `int` or `float` or `dict` or `list` or `tuple`
            The new `selected_values` value.
        type_value : `str`, optional
            The trait event type.
        """
        if self._render_function is not None:
            change_dict = {'type': 'change', 'old': old_value,
                           'name': type_value, 'new': new_value,
                           'owner': self.__str__()}
            self._render_function(change_dict)


class LinearModelParametersWidget(MenpoWidget):
    r"""
    Creates a widget for selecting parameters values when visualizing a linear
    model (e.g. PCA model).

    Note that:

    * To update the state of the widget, please refer to the
      :meth:`set_widget_state` method.
    * The selected values are stored in the ``self.selected_values`` `trait`
      which is a `list`.
    * To set the styling of this widget please refer to the
      :meth:`predefined_style` method.
    * To update the handler callback functions of the widget, please refer to
      the :meth:`replace_render_function` and :meth:`replace_variance_function`
      methods.

    Parameters
    ----------
    n_parameters : `int`
        The `list` of initial parameters values.
    render_function : `callable` or ``None``, optional
        The render function that is executed when a widgets' value changes.
        It must have signature ``render_function(change)`` where ``change`` is
        a `dict` with the following keys:

        * ``type`` : The type of notification (normally ``'change'``).
        * ``owner`` : the `HasTraits` instance
        * ``old`` : the old value of the modified trait attribute
        * ``new`` : the new value of the modified trait attribute
        * ``name`` : the name of the modified trait attribute.

        If ``None``, then nothing is assigned.
    mode : ``{'single', 'multiple'}``, optional
        If ``'single'``, only a single slider is constructed along with a
        dropdown menu that allows the parameter selection.
        If ``'multiple'``, a slider is constructed for each parameter.
    params_str : `str`, optional
        The string that will be used as description of the slider(s). The final
        description has the form ``"{}{}".format(params_str, p)``, where ``p``
        is the parameter number.
    params_bounds : (`float`, `float`), optional
        The minimum and maximum bounds, in std units, for the sliders.
    params_step : `float`, optional
        The step, in std units, of the sliders.
    plot_variance_visible : `bool`, optional
        Defines whether the button for plotting the variance will be visible
        upon construction.
    plot_variance_function : `callable` or ``None``, optional
        The plot function that is executed when the plot variance button is
        clicked. If ``None``, then nothing is assigned.
    animation_visible : `bool`, optional
        Defines whether the animation options will be visible.
    loop_enabled : `bool`, optional
        If ``True``, then the repeat mode of the animation is enabled.
    interval : `float`, optional
        The interval between the animation progress in seconds.
    interval_step : `float`, optional
        The interval step (in seconds) that is applied when fast
        forward/backward buttons are pressed.
    animation_step : `float`, optional
        The parameters step that is applied when animation is enabled.
    style : `str` (see below), optional
        Sets a predefined style at the widget. Possible options are:

            ============= ==================
            Style         Description
            ============= ==================
            ``'success'`` Green-based style
            ``'info'``    Blue-based style
            ``'warning'`` Yellow-based style
            ``'danger'``  Red-based style
            ``''``        No style
            ============= ==================

    continuous_update : `bool`, optional
        If ``True``, then the render function is called while moving a
        slider's handle. If ``False``, then the the function is called only
        when the handle (mouse click) is released.

    Example
    -------
    Let's create a linear model parameters values widget and then update its
    state. Firstly, we need to import it:

        >>> from menpowidgets.options import LinearModelParametersWidget

    Now let's define a render function that will get called on every widget
    change and will dynamically print the selected parameters:

        >>> from menpo.visualize import print_dynamic
        >>> def render_function(change):
        >>>     s = "Selected parameters: {}".format(wid.selected_values)
        >>>     print_dynamic(s)

    Create the widget with some initial options and display it:

        >>> wid = LinearModelParametersWidget(n_parameters=5,
        >>>                                   render_function=render_function,
        >>>                                   params_str='Parameter ',
        >>>                                   mode='multiple',
        >>>                                   params_bounds=(-3., 3.),
        >>>                                   plot_variance_visible=True,
        >>>                                   style='info')
        >>> wid

    By moving the sliders, the printed message gets updated. Finally, let's
    change the widget status with a new set of options:

        >>> wid.set_widget_state(n_parameters=10, params_str='',
        >>>                      params_step=0.1, params_bounds=(-10, 10),
        >>>                      plot_variance_visible=False,
        >>>                      allow_callback=True)
    """
    def __init__(self, n_parameters, render_function=None, mode='multiple',
                 params_str='Parameter ', params_bounds=(-3., 3.),
                 params_step=0.1, plot_variance_visible=True,
                 plot_variance_function=None, animation_visible=True,
                 loop_enabled=False, interval=0., interval_step=0.05,
                 animation_step=0.5, style='', continuous_update=False):

        # Get the kernel to use it later in order to make sure that the widgets'
        # traits changes are passed during a while-loop
        self.kernel = get_ipython().kernel

        # If only one slider requested, then set mode to multiple
        if n_parameters == 1:
            mode = 'multiple'

        # Create children
        if mode == 'multiple':
            self.sliders = []
            self.parameters_children = []
            for p in range(n_parameters):
                slider_title = ipywidgets.HTML(
                    value="{}{}".format(params_str, p))
                slider_wid = ipywidgets.FloatSlider(
                    description='', min=params_bounds[0], max=params_bounds[1],
                    step=params_step, value=0.,
                    continuous_update=continuous_update,
                    layout=ipywidgets.Layout(width='8cm'))
                tmp = ipywidgets.HBox([slider_title, slider_wid])
                tmp.layout.align_items = 'center'
                self.sliders.append(slider_wid)
                self.parameters_children.append(tmp)
            self.parameters_wid = ipywidgets.VBox(self.parameters_children)
            self.parameters_wid.layout.align_items = 'flex-end'
        else:
            vals = OrderedDict()
            for p in range(n_parameters):
                vals["{}{}".format(params_str, p)] = p
            self.slider = ipywidgets.FloatSlider(
                description='', min=params_bounds[0], max=params_bounds[1],
                step=params_step, value=0., readout=True,
                layout=ipywidgets.Layout(width='8cm'),
                continuous_update=continuous_update)
            self.dropdown_params = ipywidgets.Dropdown(
                options=vals, layout=ipywidgets.Layout(width='3cm'))
            self.dropdown_params.layout.margin = '0px 10px 0px 0px'
            self.parameters_wid = ipywidgets.HBox([self.dropdown_params,
                                                   self.slider])
        self.parameters_wid.layout.margin = '0px 0px 10px 0px'
        self.plot_button = ipywidgets.Button(
            description='Variance', layout=ipywidgets.Layout(width='80px'))
        self.plot_button.layout.display = (
            'inline' if plot_variance_visible else 'none')
        self.reset_button = ipywidgets.Button(
            description='Reset', layout=ipywidgets.Layout(width='80px'))
        self.plot_and_reset = ipywidgets.HBox([self.reset_button,
                                               self.plot_button])
        self.play_button = ipywidgets.Button(
            icon='play', description='', tooltip='Play animation',
            layout=ipywidgets.Layout(width='40px'))
        self.stop_button = ipywidgets.Button(
            icon='stop', description='', tooltip='Stop animation',
            layout=ipywidgets.Layout(width='40px'))
        self.fast_forward_button = ipywidgets.Button(
            icon='fast-forward', description='',
            layout=ipywidgets.Layout(width='40px'),
            tooltip='Increase animation speed')
        self.fast_backward_button = ipywidgets.Button(
            icon='fast-backward', description='',
            layout=ipywidgets.Layout(width='40px'),
            tooltip='Decrease animation speed')
        loop_icon = 'repeat' if loop_enabled else 'long-arrow-right'
        self.loop_toggle = ipywidgets.ToggleButton(
            icon=loop_icon, description='', value=loop_enabled,
            layout=ipywidgets.Layout(width='40px'), tooltip='Repeat animation')
        self.animation_buttons = ipywidgets.HBox(
            [self.play_button, self.stop_button, self.loop_toggle,
             self.fast_backward_button, self.fast_forward_button])
        self.animation_buttons.layout.display = (
            'flex' if animation_visible else 'none')
        self.animation_buttons.layout.margin = '0px 15px 0px 0px'
        self.buttons_box = ipywidgets.HBox([self.animation_buttons,
                                            self.plot_and_reset])
        self.container = ipywidgets.VBox([self.parameters_wid,
                                          self.buttons_box])

        # Create final widget
        super(LinearModelParametersWidget, self).__init__(
            [self.container], List, [0.] * n_parameters,
            render_function=render_function)

        # Assign output
        self.n_parameters = n_parameters
        self.mode = mode
        self.params_str = params_str
        self.params_bounds = params_bounds
        self.params_step = params_step
        self.plot_variance_visible = plot_variance_visible
        self.loop_enabled = loop_enabled
        self.continuous_update = continuous_update
        self.interval = interval
        self.interval_step = interval_step
        self.animation_step = animation_step
        self.animation_visible = animation_visible
        self.please_stop = False

        # Set style
        self.predefined_style(style)

        # Set functionality
        if mode == 'single':
            # Assign slider value to parameters values list
            def save_slider_value(change):
                current_parameters = list(self.selected_values)
                current_parameters[self.dropdown_params.value] = change['new']
                self.selected_values = current_parameters
            self.slider.observe(save_slider_value, names='value', type='change')

            # Set correct value to slider when drop down menu value changes
            def set_slider_value(change):
                # Temporarily remove render callback
                render_fun = self._render_function
                self.remove_render_function()
                # Set slider value
                self.slider.value = self.selected_values[change['new']]
                # Re-assign render callback
                self.add_render_function(render_fun)
            self.dropdown_params.observe(set_slider_value, names='value',
                                         type='change')
        else:
            # Assign saving values and main plotting function to all sliders
            for w in self.sliders:
                w.observe(self._save_slider_value_from_id, names='value',
                          type='change')

        def reset_parameters(name):
            # Keep old value
            old_value = self.selected_values

            # Temporarily remove render callback
            render_fun = self._render_function
            self.remove_render_function()

            # Set parameters to 0
            self.selected_values = [0.0] * self.n_parameters
            if mode == 'multiple':
                for ww in self.sliders:
                    ww.value = 0.
            else:
                self.parameters_wid.children[0].value = 0
                self.parameters_wid.children[1].value = 0.

            # Re-assign render callback and trigger it
            self.add_render_function(render_fun)
            self.call_render_function(old_value, self.selected_values)
        self.reset_button.on_click(reset_parameters)

        # Set functionality
        def loop_pressed(change):
            if change['new']:
                self.loop_toggle.icon = 'repeat'
            else:
                self.loop_toggle.icon = 'long-arrow-right'
            self.kernel.do_one_iteration()
        self.loop_toggle.observe(loop_pressed, names='value', type='change')

        def fast_forward_pressed(name):
            tmp = self.interval
            tmp -= self.interval_step
            if tmp < 0:
                tmp = 0
            self.interval = tmp
            self.kernel.do_one_iteration()
        self.fast_forward_button.on_click(fast_forward_pressed)

        def fast_backward_pressed(name):
            self.interval += self.interval_step
            self.kernel.do_one_iteration()
        self.fast_backward_button.on_click(fast_backward_pressed)

        def animate(change):
            reset_parameters('')
            self.please_stop = False
            self.reset_button.disabled = True
            self.plot_button.disabled = True
            if mode == 'multiple':
                n_sliders = self.n_parameters
                slider_id = 0
                while slider_id < n_sliders:
                    # animate from 0 to min
                    slider_val = 0.
                    while slider_val > self.params_bounds[0]:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val -= self.animation_step

                        # set value
                        self.sliders[slider_id].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # animate from min to max
                    slider_val = self.params_bounds[0]
                    while slider_val < self.params_bounds[1]:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val += self.animation_step

                        # set value
                        self.sliders[slider_id].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # animate from max to 0
                    slider_val = self.params_bounds[1]
                    while slider_val > 0.:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val -= self.animation_step

                        # set value
                        self.sliders[slider_id].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # reset value
                    self.sliders[slider_id].value = 0.

                    # Check stop flag
                    if self.please_stop:
                        break

                    # update slider id
                    if self.loop_toggle.value and slider_id == n_sliders - 1:
                        slider_id = 0
                    else:
                        slider_id += 1

                if not self.loop_toggle.value and slider_id >= n_sliders:
                    self.stop_animation()
            else:
                n_sliders = self.n_parameters
                slider_id = 0
                self.please_stop = False
                while slider_id < n_sliders:
                    # set dropdown value
                    self.parameters_wid.children[0].value = slider_id

                    # animate from 0 to min
                    slider_val = 0.
                    while slider_val > self.params_bounds[0]:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val -= self.animation_step

                        # set value
                        self.parameters_wid.children[1].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # animate from min to max
                    slider_val = self.params_bounds[0]
                    while slider_val < self.params_bounds[1]:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val += self.animation_step

                        # set value
                        self.parameters_wid.children[1].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # animate from max to 0
                    slider_val = self.params_bounds[1]
                    while slider_val > 0.:
                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                        # Check stop flag
                        if self.please_stop:
                            break

                        # update slider value
                        slider_val -= self.animation_step

                        # set value
                        self.parameters_wid.children[1].value = slider_val

                        # wait
                        sleep(self.interval)

                        # Run IPython iteration.
                        self.kernel.do_one_iteration()

                    # reset value
                    self.parameters_wid.children[1].value = 0.

                    # Check stop flag
                    if self.please_stop:
                        break

                    # update slider id
                    if self.loop_toggle.value and slider_id == n_sliders - 1:
                        slider_id = 0
                    else:
                        slider_id += 1
            self.reset_button.disabled = False
            self.plot_button.disabled = False
        self.play_button.on_click(animate)

        def stop_pressed(_):
            self.stop_animation()
        self.stop_button.on_click(stop_pressed)

        # Set plot variance function
        self._variance_function = None
        self.add_variance_function(plot_variance_function)

    def _save_slider_value_from_id(self, change):
        current_parameters = list(self.selected_values)
        i = self.sliders.index(change['owner'])
        current_parameters[i] = change['new']
        self.selected_values = current_parameters

    def predefined_style(self, style):
        r"""
        Function that sets a predefined style on the widget.

        Parameters
        ----------
        style : `str` (see below)
            Style options:

                ============= ==================
                Style         Description
                ============= ==================
                ``'success'`` Green-based style
                ``'info'``    Blue-based style
                ``'warning'`` Yellow-based style
                ``'danger'``  Red-based style
                ``''``        No style
                ============= ==================
        """
        self.container.box_style = style
        self.container.border = '0px'
        self.play_button.button_style = 'success'
        self.stop_button.button_style = 'danger'
        self.fast_forward_button.button_style = 'info'
        self.fast_backward_button.button_style = 'info'
        self.loop_toggle.button_style = 'warning'
        self.reset_button.button_style = 'danger'
        self.plot_button.button_style = 'primary'

    def stop_animation(self):
        r"""
        Method that stops an active annotation.
        """
        self.please_stop = True

    def add_variance_function(self, variance_function):
        r"""
        Method that adds a `variance_function()` to the `Variance` button of the
        widget. The given function is also stored in `self._variance_function`.

        Parameters
        ----------
        variance_function : `callable` or ``None``, optional
            The variance function that behaves as a callback. If ``None``,
            then nothing is added.
        """
        self._variance_function = variance_function
        if self._variance_function is not None:
            self.plot_button.on_click(self._variance_function)

    def remove_variance_function(self):
        r"""
        Method that removes the current `self._variance_function()` from
        the `Variance` button of the widget and sets
        ``self._variance_function = None``.
        """
        self.plot_button.on_click(self._variance_function, remove=True)
        self._variance_function = None

    def replace_variance_function(self, variance_function):
        r"""
        Method that replaces the current `self._variance_function()` of the
        `Variance` button of the widget with the given `variance_function()`.

        Parameters
        ----------
        variance_function : `callable` or ``None``, optional
            The variance function that behaves as a callback. If ``None``,
            then nothing happens.
        """
        # remove old function
        self.remove_variance_function()

        # add new function
        self.add_variance_function(variance_function)

    def set_widget_state(self, n_parameters=None, params_str=None,
                         params_bounds=None, params_step=None,
                         plot_variance_visible=True, animation_step=0.5,
                         allow_callback=True):
        r"""
        Method that updates the state of the widget with a new set of options.

        Parameters
        ----------
        n_parameters : `int`
            The `list` of initial parameters values.
        params_str : `str`, optional
            The string that will be used as description of the slider(s). The
            final description has the form ``"{}{}".format(params_str, p)``,
            where ``p`` is the parameter number.
        params_bounds : (`float`, `float`), optional
            The minimum and maximum bounds, in std units, for the sliders.
        params_step : `float`, optional
            The step, in std units, of the sliders.
        plot_variance_visible : `bool`, optional
            Defines whether the button for plotting the variance will be visible
            upon construction.
        animation_step : `float`, optional
            The parameters step that is applied when animation is enabled.
        allow_callback : `bool`, optional
            If ``True``, it allows triggering of any callback functions.
        """
        # Keep old value
        old_value = self.selected_values

        # Temporarily remove render callback
        render_function = self._render_function
        self.remove_render_function()

        # Parse given options
        if n_parameters is None:
            n_parameters = self.n_parameters
        if params_str is None:
            params_str = ''
        if params_bounds is None:
            params_bounds = self.params_bounds
        if params_step is None:
            params_step = self.params_step

        # Set plot variance visibility
        self.plot_button.layout.visibility = (
            'visible' if plot_variance_visible else 'hidden')
        self.animation_step = animation_step

        # Update widget
        if n_parameters == self.n_parameters:
            # The number of parameters hasn't changed
            if self.mode == 'multiple':
                for p, sl in enumerate(self.sliders):
                    self.parameters_children[p].children[0].value = \
                        "{}{}".format(params_str, p)
                    sl.min = params_bounds[0]
                    sl.max = params_bounds[1]
                    sl.step = params_step
            else:
                self.slider.min = params_bounds[0]
                self.slider.max = params_bounds[1]
                self.slider.step = params_step
                if not params_str == '':
                    vals = OrderedDict()
                    for p in range(n_parameters):
                        vals["{}{}".format(params_str, p)] = p
                    self.dropdown_params.options = vals
        else:
            # The number of parameters has changed
            self.selected_values = [0.] * n_parameters
            if self.mode == 'multiple':
                # Create new sliders
                self.sliders = []
                self.parameters_children = []
                for p in range(n_parameters):
                    slider_title = ipywidgets.HTML(
                        value="{}{}".format(params_str, p))
                    slider_wid = ipywidgets.FloatSlider(
                        description='', min=params_bounds[0],
                        max=params_bounds[1],
                        step=params_step, value=0., width='8cm',
                        continuous_update=self.continuous_update)
                    tmp = ipywidgets.HBox([slider_title, slider_wid])
                    tmp.layout.align_items = 'center'
                    self.sliders.append(slider_wid)
                    self.parameters_children.append(tmp)
                self.parameters_wid.children = self.parameters_children

                # Assign saving values and main plotting function to all sliders
                for w in self.sliders:
                    w.observe(self._save_slider_value_from_id, names='value',
                              type='change')
            else:
                self.slider.min = params_bounds[0]
                self.slider.max = params_bounds[1]
                self.slider.step = params_step
                vals = OrderedDict()
                for p in range(n_parameters):
                    vals["{}{}".format(params_str, p)] = p
                if self.dropdown_params.value == 0 and n_parameters > 1:
                    self.dropdown_params.value = 1
                self.dropdown_params.value = 0
                self.dropdown_params.options = vals
                self.slider.value = 0.

        # Re-assign render callback
        self.add_render_function(render_function)

        # Assign new selected options
        self.n_parameters = n_parameters
        self.params_str = params_str
        self.params_bounds = params_bounds
        self.params_step = params_step
        self.plot_variance_visible = plot_variance_visible

        # trigger render function if allowed
        if allow_callback:
            self.call_render_function(old_value, self.selected_values)
