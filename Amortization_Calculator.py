print("This function calculates and prints and ammortization table")

r = float(input("What is you interest rate? "))
r = r/100
i = r/12


n = float(input("How long is your Loan? "))
n=n*12

peroid = int(n) -1
p = float(input("What is your Priciple? "))
interest = i

r =float(r) ##
n =float(n) ##
p =float(p) ##
i =float(i) ##

interestPaid  = 0
principalPaid = 0 
remainingPrincipal = 0
totalInterestPaid = 0
totalPrincipalPaid = 0
paymentPeroid = 1

Payment_Table = {
    'Payment Peroid' : paymentPeroid,
    'Remaining Principle' : remainingPrincipal,
    'Principle Paid': principalPaid,
    'Interest Paid' : totalInterestPaid,
    'Total Principle Paid' : totalPrincipalPaid,
    }

monthlyPayment = (p*(i*(1+i)**n))/(((1+i)**n)-1)
interestPaid  = p*interest
principalPaid = monthlyPayment - interestPaid
remainingPrincipal = p - (principalPaid - remainingPrincipal)
totalInterestPaid = totalInterestPaid + interestPaid
totalPrincipalPaid = principalPaid + totalPrincipalPaid

Payment_Table['Payment Peroid'] = paymentPeroid
Payment_Table['Remaining Principle'] = remainingPrincipal
Payment_Table['Principle Paid'] = principalPaid
Payment_Table['Interest Paid'] = totalInterestPaid
Payment_Table['Total Principle Paid'] =totalPrincipalPaid

print(Payment_Table)

for index in range(0,peroid):
    
    monthlyPayment = (p*(i*(1+i)**n))/(((1+i)**n)-1)
    interestPaid  = remainingPrincipal*interest
    principalPaid = monthlyPayment - interestPaid
    remainingPrincipal = remainingPrincipal -principalPaid
    totalInterestPaid = totalInterestPaid + interestPaid
    totalPrincipalPaid = principalPaid + totalPrincipalPaid
    paymentPeroid = paymentPeroid +1

    Payment_Table['Payment Peroid'] = paymentPeroid
    Payment_Table['Remaining Principle'] = remainingPrincipal
    Payment_Table['Principle Paid'] = principalPaid
    Payment_Table['Interest Paid'] = totalInterestPaid
    Payment_Table['Total Principle Paid'] =totalPrincipalPaid


    print("\n",Payment_Table)

print("\nFor a loan of $", p, " at an interest rate of",r*100, "your loan will have", paymentPeroid,"payment peroids of $",monthlyPayment, " monthly.")

##
##
exit = input("Press an key to exit")
