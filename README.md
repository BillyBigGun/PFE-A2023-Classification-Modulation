# Git commands
```bash
git clone <repo>
git add .                   #add all the changes you made
git commit -m "<message>"   #the message that represent the changes made
git push origin <branch>    #the name of the branch you want to push to
git pull                    #pull the changes online to your local device on your current branch
git branch <branch>         #the name of the branch you want ot create
git checkout <branch>       #the branch you want to work on
```

# Cloning 
The project contains submodules. Clone it using this command:
```bash
git clone --recurse-submodules <repository-url>
```

# Using ThinkDSP
To use the classes in ThinkDSP, import it to you script with:
```python
from lib.ThinkDSP.code.thinkdsp import Wave
```
We imported wave as an example, but change wave for the class you want to import

# UML drawings
## Waveform
![UML Diagram Waveform](./docs/UML-waveform.png)

## Neural Network
![UML Diagram NN](./docs/UML-NN.png)
