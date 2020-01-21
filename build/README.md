# Setting up your environment

These directions assume you are working from a fresh Ubuntu 18.04 EC2 instance.

## Setting up X11 forwarding
When you ssh into your instance, make sure to enable the <pre>-Y</pre> flag.
Next, edit your ssh_config file:

    sudo vim  /etc/ssh/ssh_config

Add the line <pre> ForwardX11 yes </pre>.
Next, restart your ssh service:

    sudo service ssh restart

Finally, install some X11 tools and test that your forwarding works with <pre>xclock</pre>:

    sudo apt install x11-apps
    xlock

If you set it up right, a little clock will appear on your screen!
