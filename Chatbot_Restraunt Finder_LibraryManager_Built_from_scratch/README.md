<img width=400 height=400 src="results.png"/>

Simple Chatbot from library management and Restraunt Finding 
built from scratch using Python.


Core concepts:

Intents (List of possible phrases Ex: Can you order me something)
Entities (Objects/Actions Ex: Restraunt)
Actions (Some action on Db Ex: Booking Restraunt)


--------------------------------------------------------


Dataset Used: CSV files

Both DBs have more than 100 entries each.

Library (booklibrary.csv)

Restaurant (Restaurantlist.csv)

--------------------------------------------------------

Number of intents (2)

*Library (200+ )
*Restaurant (150+)

--------------------------------------------------------

**Number of entities are more than 3 for each problem.

Problem Domain: Library
--------------------------------------------------------

	The system takes the Name, Author, Subject (Genre) of the book and Registration number as inputs from the user (like userid). If a book matching the details entered exists in the database, the system returns a message "FetchBook task is completed" along with the book details and Invoice number. It returns a message indicating that the book not found if the details given by the user does not match any record in the database.

	The user need to provide the exact Name, Author Name and Genre of the book. Registration number can be any integer value.

--------------------------------------------------------
Problem Domain: Restaurant
	
	The system takes the Location (North,South,East,West), Cuisine(Chinese….etc), CostType(Cheap,Medium, Expensive,….etc) and suggests the restaurants names to user. It shows multiple restraints if it has multiple matches


--------------------------------------------------------
Additional Notes

	We have used regular expressions to correct typos.


