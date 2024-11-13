// Check if browser supports notifications
if (!('Notification' in window)) {
    console.log("Notifications not supported");
  }
  
  // Request permission from user
  Notification.requestPermission().then(function(result) {
    if (result === 'granted') {
      console.log("Notification permission granted");
    }
  });

// Create a notification
const notification = new Notification("Notification title", {
    body: "Notification body text",
    icon: "path/to/icon.png"
  });
  
  // Close the notification after 10 seconds
  setTimeout(() => {
    notification.close(); 
  }, 10 * 1000);
  