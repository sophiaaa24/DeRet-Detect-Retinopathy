<?php require_once "logic.php"; ?>
<html>
<head>
    <title>Login</title>
    <link rel="stylesheet" href="css/styles.css">
</head>
<body>
    <div class="card">
        <div class="card-body">
            <?php if(count($errors) > 0) { ?>
                <div class="alert">
                    <?php
                        foreach($errors as $showerror){
                            echo $showerror;
                        }
                    ?>
                </div>
            <?php } ?>
            <form action="login-user.php" method="POST">
                <h2 class="title">Login</h2>
                <p class="subtitle">Login with your registered Email and Password.</p>
                    <div>
                        <input type="email" name="email" placeholder="Email Address" class="input_css" pattern="[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$" required>
                    </div>
                    <div>
                        <input type="password" name="password" placeholder="Password" class="input_css" pattern="^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$" required>
                    </div>
                   <div class="forgot-password"><a href="forgot-password.php">Forgot Password?</a></div><br><br>
                    <div>
                        <input type="submit" name="login" value="Login" class="submit_css">
                    </div>
                    <?php if (!empty($error)) { ?>
                            <div class="error"><?php echo $error; ?></div>
                    <?php } ?>
                    <br><div class="signup-link">Not registered yet? <a href="signup-user.php">SignUp Now</a></div>
            </form>
        </div>
    </div>
</body>
</html>
