import sys
import smtplib
#from email.mime.multipart import MIMEMultipart
#from email.mime.text import MIMEText
from email.MIMEBase import MIMEBase
from email.MIMEText import MIMEText
from email.MIMEMultipart import MIMEMultipart
from email import Encoders

class Email(object):
    
    # @staticmethod
    # def checkEmailName(customer):
    #     supported=['golden', "samuel"]
    #     if customer in supported:
    #         return True, supported
    
    #     return False, supported

    @staticmethod
    def sendEmail(message_txt, attachments, subject):
        
        faceTeamEmailGroup =['shenfei0427@thundersoft.com',  
                             'lujx0301@thundersoft.com',     
                             'xieyt1029@thundersoft.com'] 
        #Use the Face Team account to send email
        sender = 'hzface0704@thundersoft.com'
        userName = 'hzface0704'
        password = 'hzface0704'        
        receiver = ';'.join(faceTeamEmailGroup)
        
        try:
            #link the email server 
            server = smtplib.SMTP('mail.thundersoft.com')
            server.ehlo()
            server.starttls() # use encript mode
            server.ehlo() 
            server.login(userName, password)
        
            text=MIMEText(message_txt)
            message = MIMEMultipart()  
            message["Subject"] = subject
            message['From']=sender
            message['To'] =receiver
            message.attach(text)

            #Add attachments
            for f in attachments:
                if not os.path.exists(f):
                    continue
                part = MIMEBase('application', "octet-stream")
                part.set_payload( open(f,"rb").read() )
                Encoders.encode_base64(part)
                part.add_header('Content-Disposition', 
                                'attachment; filename="%s"'%os.path.basename(f))
                message.attach(part) 
                
        
            server.sendmail(sender, receiver , message.as_string())
            server.quit()
        except:
            print("Send Email Failed")
