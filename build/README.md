# Setting up your environment

These directions assume you are working from a fresh Ubuntu 18.04 EC2 instance.
I'm including some optional directions for X11 forwarding; if you are working on a local machine, this isn't needed.

## Setting up X11 forwarding (optional)
When you ssh into your instance, make sure to enable the **-Y** flag.
Next, edit your ssh_config file:

    sudo vim  /etc/ssh/ssh_config

Add the line **ForwardX11 yes** under **Host** <b>*</b>.

Note that you will get an X11 forwarding error when connecting to services that don't need X11 (such as github). 
Configure your **/etc/ssh/ssh_config** file to not forward to certain servers like so:
<pre>
Host github.com
    ForwardX11 no 
</pre>

Next, restart your ssh service:

    sudo service ssh restart

Finally, install some X11 tools and test that your forwarding works with **xclock**:

    sudo apt install x11-apps
    xclock

If you set it up right, a little clock will appear on your screen!

## Installing apt-get packages
The script **apt_get.sh** includes sudo commands to install all the necessary systemwide tools.
After that, you can use **requirements.txt** to install the needed python packages to your pip3.
I would recommend setting up a virtual environment first (the second and third lines):

    source build/apt_get.sh
    python3 -m venv build/my_first_env
    source build/my_first_env/bin/activate
    pip3 install build/requirements.txt

