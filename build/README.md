## Setting up X11 forwarding (optional)
The following directions will allow you to view images stored on a remote machine through your local display with X11 forwarding.

If you are working entirely on your local machine or plan to download images to your local machine, this isn't needed.

When you ssh into your instance, make sure to enable the **-Y** flag.
Edit your sshd_config file:
    
    sudo vim /etc/ssh/sshd_config

Uncomment the lines **ForwardX11 yes** and **X11UseLocalhost no**.

Next, logout and log back in for your changes to take effect. 

Finally, install some X11 tools and test that your forwarding works with **xclock**:

    sudo apt install x11-apps
    xclock

If you set it up right, a little clock will appear on your screen!

This will allow you to view images on a remote machine through your local display.
