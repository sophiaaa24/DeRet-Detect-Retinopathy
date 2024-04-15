<?php require_once "logic.php"; ?>
<html>
<head>
    <title>Sign Up</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
            <div class="card">
                <div class="card-body">
                    <form action="signup-user.php" method="POST">
                        <h2 class="title">Sign Up</h2>
                        <?php
                    if(count($errors) == 1){
                        ?>
                        <div class="alert">
                            <?php
                            foreach($errors as $showerror){
                                echo $showerror;
                            }
                            ?>
                        </div>
                        <?php
                    }elseif(count($errors) > 1){
                        ?>
                        <div class="alert">
                            <?php
                            foreach($errors as $showerror){
                                ?>
                                <li><?php echo $showerror; ?></li>
                                <?php
                            }
                            ?>
                        </div>
                        <?php
                    }
                    ?>
                        <div>
                            <input type="text" name="name" id="name" placeholder="Full Name" class="input_css" required>
                        </div>
                        <div>
                            <input type="email" name="email" id="email" placeholder="Email Address" class="input_css" pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$" required>
                        </div>
                        <div>
                            <input type="password" name="password" id="password" placeholder="Password" class="input_css" pattern="^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" required>
                        </div>
                        <div >
                        <input class="input_css" type="password" name="cpassword" placeholder="Confirm password" pattern="^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" required>
                        </div>
                    
                        <div>
                            <input type="submit" name="signup" value="Sign Up" class="submit_css">
                        </div>
                        <?php if (!empty($error)) { ?>
                            <div class="error"><?php echo $error; ?></div>
                        <?php } ?>
                        <br><div class="login-link">Already have an account? <a href="login-user.php">Login Here</a></div>
                    </form>
                </div>
            </div>
                        
</body>
</html>
